"""Render images of a target through a known camera.

Used for the test suite (the true camera is known, so results can be checked)
and for the built-in sample images. Rendering maps every output pixel back
through the lens model onto the board plane (inverse mapping), so lens
distortion is exact everywhere, not only at the board's corners.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .targets import CHARUCO, Target

PX_PER_MM = 8.0


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray


def default_camera() -> Camera:
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 480.0], [0, 0, 1]], np.float64)
    dist = np.array([-0.12, 0.10, 0.0005, -0.0003, 0.0], np.float64)
    return Camera(1280, 960, K, dist)


def board_texture(target: Target, margin_squares: float = 1.0) -> tuple[np.ndarray, float]:
    """Flat image of the target with a white margin. Returns (image, margin in mm)."""
    margin_mm = margin_squares * target.square_mm
    w = int(round(target.width_mm * PX_PER_MM))
    h = int(round(target.height_mm * PX_PER_MM))
    m = int(round(margin_mm * PX_PER_MM))
    if target.kind == CHARUCO:
        inner = target.charuco_board().generateImage((w, h), marginSize=0)
    else:
        inner = np.full((h, w), 255, np.uint8)
        sq = target.square_mm * PX_PER_MM
        for j in range(target.squares_y):
            for i in range(target.squares_x):
                if (i + j) % 2 == 0:
                    x0, y0 = int(round(i * sq)), int(round(j * sq))
                    x1, y1 = int(round((i + 1) * sq)), int(round((j + 1) * sq))
                    inner[y0:y1, x0:x1] = 0
    tex = np.full((h + 2 * m, w + 2 * m), 255, np.uint8)
    tex[m : m + h, m : m + w] = inner
    return tex, margin_mm


def render(
    target: Target,
    camera: Camera,
    rvec: np.ndarray,
    tvec: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_sigma: float = 2.0,
    blur: float = 0.8,
    background: int = 120,
) -> np.ndarray:
    """Render the target at pose (rvec, tvec), board frame to camera frame, mm."""
    tex, margin_mm = board_texture(target)
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    t = np.asarray(tvec, np.float64).reshape(3)
    # The texture is finer than the output image; blur it first so that
    # bilinear sampling does not alias.
    minification = PX_PER_MM * float(t[2]) / float(camera.K[0, 0])
    if minification > 1.0:
        tex = cv2.GaussianBlur(tex, (0, 0), 0.5 * minification)

    u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64),
                       np.arange(camera.height, dtype=np.float64))
    px = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(px, camera.K, camera.dist).reshape(-1, 2)
    d = np.hstack([rays, np.ones((len(rays), 1))])  # ray directions, camera frame

    # Board plane: points X = R p + t with p_z = 0. Solve for p along each ray.
    n = R[:, 2]
    denom = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (t @ n) / denom
    P = d * s[:, None]
    p = (P - t) @ R  # R^T (P - t)
    map_x = ((p[:, 0] + margin_mm) * PX_PER_MM - 0.5).astype(np.float32)
    map_y = ((p[:, 1] + margin_mm) * PX_PER_MM - 0.5).astype(np.float32)
    bad = ~np.isfinite(s) | (s <= 0)
    map_x[bad] = -10
    map_y[bad] = -10

    img = cv2.remap(
        tex,
        map_x.reshape(camera.height, camera.width),
        map_y.reshape(camera.height, camera.width),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if rng is not None and noise_sigma > 0:
        noisy = img.astype(np.float32) + rng.normal(0, noise_sigma, img.shape)
        img = np.clip(noisy, 0, 255).astype(np.uint8)
    return img


def sample_poses(
    target: Target, camera: Camera, count: int, seed: int = 7
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Poses that spread the board over the image at varied tilts and distances."""
    rng = np.random.default_rng(seed)
    f = float(camera.K[0, 0])
    centre = np.array([target.width_mm / 2, target.height_mm / 2, 0.0])
    poses = []
    for i in range(count):
        # Distance at which the board fills 45 to 80 percent of the image width.
        fill = rng.uniform(0.45, 0.8)
        z = f * target.width_mm / (fill * camera.width)
        tilt_x = np.deg2rad(rng.uniform(-35, 35))
        tilt_y = np.deg2rad(rng.uniform(-35, 35))
        roll = np.deg2rad(rng.uniform(-20, 20))
        if i % 6 == 0:
            tilt_x = tilt_y = 0.0  # a few frontal views
        Rx, _ = cv2.Rodrigues(np.array([tilt_x, 0, 0]))
        Ry, _ = cv2.Rodrigues(np.array([0, tilt_y, 0]))
        Rz, _ = cv2.Rodrigues(np.array([0, 0, roll]))
        R = Rz @ Ry @ Rx
        # Where the board centre lands in the image.
        slack_x = max(0.0, (1 - fill) / 2) * camera.width
        slack_y = 0.25 * camera.height
        cu = camera.K[0, 2] + rng.uniform(-slack_x, slack_x)
        cv_ = camera.K[1, 2] + rng.uniform(-slack_y, slack_y)
        c_cam = np.array([(cu - camera.K[0, 2]) / f * z, (cv_ - camera.K[1, 2]) / f * z, z])
        t = c_cam - R @ centre
        rvec, _ = cv2.Rodrigues(R)
        poses.append((rvec.reshape(3), t))
    return poses


def render_set(
    target: Target, camera: Camera | None = None, count: int = 24, seed: int = 7
) -> list[np.ndarray]:
    camera = camera or default_camera()
    rng = np.random.default_rng(seed + 1000)
    return [render(target, camera, r, t, rng) for r, t in sample_poses(target, camera, count, seed)]


def true_corners(target: Target, camera: Camera, rvec, tvec) -> np.ndarray:
    """Where the inner corners project for a pose, (N, 2)."""
    proj, _ = cv2.projectPoints(target.object_points().astype(np.float64), rvec, tvec, camera.K, camera.dist)
    return proj.reshape(-1, 2)
