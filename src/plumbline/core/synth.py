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

from . import lens
from .targets import APRILGRID, APRILGRID_BORDER_BITS, CHARUCO, Target

PX_PER_MM = 8.0


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    fisheye: bool = False


def default_camera() -> Camera:
    K = np.array([[1500.0, 0, 640.0], [0, 1500.0, 480.0], [0, 0, 1]], np.float64)
    dist = np.array([-0.12, 0.10, 0.0005, -0.0003, 0.0], np.float64)
    return Camera(1280, 960, K, dist)


def fisheye_camera() -> Camera:
    """A wide lens, about 130 degrees across the image width."""
    K = np.array([[560.0, 0, 640.0], [0, 560.0, 480.0], [0, 0, 1]], np.float64)
    dist = np.array([-0.03, 0.006, -0.002, 0.0004], np.float64)
    return Camera(1280, 960, K, dist, True)


def board_texture(target: Target, margin_squares: float = 1.0) -> tuple[np.ndarray, tuple[float, float]]:
    """Flat image of the target with a white margin.

    Returns (image, origin): where the origin of object_points() sits in the
    image, in mm from its top-left corner.
    """
    margin_mm = margin_squares * target.square_mm
    w = int(round(target.width_mm * PX_PER_MM))
    h = int(round(target.height_mm * PX_PER_MM))
    m = int(round(margin_mm * PX_PER_MM))
    origin = (margin_mm, margin_mm)
    if target.kind == CHARUCO:
        inner = target.charuco_board().generateImage((w, h), marginSize=0)
    elif target.kind == APRILGRID:
        inner = np.full((h, w), 255, np.uint8)
        side = int(round(target.square_mm * PX_PER_MM))
        dictionary = target.aruco_dictionary()
        for t in range(target.marker_count):
            x, y = target.tag_origin(t)
            tag = cv2.aruco.generateImageMarker(dictionary, target.first_marker_id + t, side,
                                                borderBits=APRILGRID_BORDER_BITS)
            x0, y0 = int(round(x * PX_PER_MM)), int(round(y * PX_PER_MM))
            inner[y0:y0 + side, x0:x0 + side] = tag[::-1, ::-1]  # turned by 180 degrees, as Kalibr prints them
    elif target.is_circles:
        r = target.circle_diameter_mm / 2
        origin = (margin_mm + r, margin_mm + r)
        inner = np.full((h, w), 255, np.uint8)
        shift = 6  # sub-pixel circle centres and radius
        for x, y, _z in target.object_points():
            cx = (x + r) * PX_PER_MM - 0.5
            cy = (y + r) * PX_PER_MM - 0.5
            cv2.circle(inner, (int(round(cx * 2**shift)), int(round(cy * 2**shift))),
                       int(round(r * PX_PER_MM * 2**shift)), 0, -1, cv2.LINE_AA, shift)
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
    return tex, origin


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
    tex, origin = board_texture(target)
    return render_texture(tex, PX_PER_MM, origin, camera, rvec, tvec,
                          rng=rng, noise_sigma=noise_sigma, blur=blur, background=background)


def render_texture(
    tex: np.ndarray,
    px_per_mm: float,
    origin_mm: tuple[float, float],
    camera: Camera,
    rvec: np.ndarray,
    tvec: np.ndarray,
    rng: np.random.Generator | None = None,
    noise_sigma: float = 2.0,
    blur: float = 0.8,
    background: int = 120,
    shading: float = 0.0,
    black: int = 0,
    white: int = 255,
) -> np.ndarray:
    """Render any flat picture of a target, for example the print file of a real board.

    origin_mm is where the pattern's origin (outer corner of the first square)
    sits in the picture, measured from the picture's top-left corner.
    shading adds uneven lighting and vignetting (0 = none, 0.4 = strong);
    black and white set the grey levels the print reaches in the image.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    t = np.asarray(tvec, np.float64).reshape(3)
    # The texture is finer than the output image; blur it first so that
    # bilinear sampling does not alias.
    minification = px_per_mm * float(t[2]) / float(camera.K[0, 0])
    if minification > 1.0:
        tex = cv2.GaussianBlur(tex, (0, 0), 0.5 * minification)

    u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64),
                       np.arange(camera.height, dtype=np.float64))
    px = np.stack([u.ravel(), v.ravel()], axis=1)
    d = lens.pixel_directions(px, camera.K, camera.dist, camera.fisheye)  # camera frame

    # Board plane: points X = R p + t with p_z = 0. Solve for p along each ray.
    n = R[:, 2]
    denom = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (t @ n) / denom
    P = d * s[:, None]
    p = (P - t) @ R  # R^T (P - t)
    map_x = ((p[:, 0] + origin_mm[0]) * px_per_mm - 0.5).astype(np.float32)
    map_y = ((p[:, 1] + origin_mm[1]) * px_per_mm - 0.5).astype(np.float32)
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
    ).astype(np.float32)
    if black != 0 or white != 255:
        img = black + img * (white - black) / 255.0
    if shading > 0:
        xx = (u - camera.width / 2) / (camera.width / 2)
        yy = (v - camera.height / 2) / (camera.height / 2)
        angle = rng.uniform(0, 2 * np.pi) if rng is not None else 0.7
        gradient = 1.0 + shading * 0.5 * (np.cos(angle) * xx + np.sin(angle) * yy)
        vignette = 1.0 - shading * 0.5 * (xx**2 + yy**2) / 2.0
        img = img * gradient * vignette
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if rng is not None and noise_sigma > 0:
        img = img + rng.normal(0, noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def sample_poses(
    target: Target, camera: Camera, count: int, seed: int = 7
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Poses that spread the board over the image at varied tilts and distances."""
    rng = np.random.default_rng(seed)
    f = float(camera.K[0, 0])
    centre = np.array([*target.centre_mm, 0.0])
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
        if camera.fisheye:
            ray = lens.pixel_directions([[cu, cv_]], camera.K, camera.dist, True)[0]
            c_cam = ray * z  # z is the distance along the ray here
        else:
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
    return lens.project(target.object_points(), rvec, tvec, camera.K, camera.dist, camera.fisheye)
