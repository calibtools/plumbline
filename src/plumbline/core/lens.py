"""The two lens models Plumbline fits, behind one set of functions.

Standard: OpenCV's pinhole model with radial, tangential and optional
rational and thin prism terms. Fisheye: OpenCV's fisheye model
(equidistant projection with four radial terms k1 to k4).
"""
from __future__ import annotations

import cv2
import numpy as np

STANDARD_NAMES = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tx", "ty"]
FISHEYE_NAMES = ["k1", "k2", "k3", "k4"]


def names(fisheye: bool, count: int) -> list[str]:
    return (FISHEYE_NAMES if fisheye else STANDARD_NAMES)[:count]


def project(obj, rvec, tvec, K, dist, fisheye: bool = False) -> np.ndarray:
    """Project board points (N, 3) into the image. Returns (N, 2)."""
    obj = np.asarray(obj, np.float64).reshape(-1, 1, 3)
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)
    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).reshape(-1)
    if fisheye:
        proj, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, dist.reshape(4, 1))
    else:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return np.asarray(proj, np.float64).reshape(-1, 2)


def pixel_directions(px, K, dist, fisheye: bool = False) -> np.ndarray:
    """Viewing direction of each pixel in the camera frame, (N, 3).

    For the standard model the directions are (x, y, 1). For the fisheye
    model they are unit vectors, which also covers pixels that look
    sideways or backwards; pixels outside what the model can describe
    come back as NaN.
    """
    px = np.asarray(px, np.float64).reshape(-1, 2)
    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).reshape(-1)
    if not fisheye:
        rays = cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        return np.hstack([rays, np.ones((len(rays), 1))])

    x = (px[:, 0] - K[0, 2]) / K[0, 0]
    y = (px[:, 1] - K[1, 2]) / K[1, 1]
    theta_d = np.hypot(x, y)
    k1, k2, k3, k4 = dist[:4]
    theta = theta_d.copy()
    for _ in range(30):  # Newton steps on theta * (1 + k1 theta^2 + ...) = theta_d
        t2 = theta * theta
        f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = theta - f / df
    t2 = theta * theta
    back = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    bad = ~np.isfinite(theta) | (theta < 0) | (theta > np.pi) | (np.abs(back - theta_d) > 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = np.where(theta_d > 0, x / theta_d, 0.0)
        uy = np.where(theta_d > 0, y / theta_d, 0.0)
    out = np.stack([np.sin(theta) * ux, np.sin(theta) * uy, np.cos(theta)], axis=1)
    out[bad] = np.nan
    return out


def solve_pose(obj, corners, K, dist, fisheye: bool = False):
    """Pose of the board from one image under a known camera. Returns (rvec, tvec) or None."""
    obj = np.asarray(obj, np.float64).reshape(-1, 3)
    corners = np.asarray(corners, np.float64).reshape(-1, 2)
    if fisheye:
        d = pixel_directions(corners, K, dist, True)
        ok = np.isfinite(d).all(axis=1) & (d[:, 2] > 1e-6)
        if ok.sum() < 4:
            return None
        flat = d[ok, :2] / d[ok, 2:3]
        done, rvec, tvec = cv2.solvePnP(obj[ok], flat, np.eye(3), np.zeros(5))
    else:
        done, rvec, tvec = cv2.solvePnP(obj, corners, np.asarray(K, np.float64), np.asarray(dist, np.float64))
    if not done:
        return None
    return np.asarray(rvec, np.float64).reshape(3), np.asarray(tvec, np.float64).reshape(3)


def corrected_view(gray: np.ndarray, corners, K, dist, fisheye: bool = False):
    """The image with lens distortion removed, and where the corners land in it.

    The corrected image keeps the size of the original and shows the part
    of the scene that stays sharp after straightening; straight edges in
    the scene come out straight.
    """
    h, w = gray.shape[:2]
    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).reshape(-1)
    corners = np.asarray(corners, np.float64).reshape(-1, 1, 2)
    if fisheye:
        D = dist.reshape(4, 1)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
        out = cv2.fisheye.undistortImage(gray, K, D, Knew=new_K, new_size=(w, h))
        moved = cv2.fisheye.undistortPoints(corners, K, D, P=new_K) if len(corners) else corners
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0.0, (w, h))
        out = cv2.undistort(gray, K, dist, None, new_K)
        moved = cv2.undistortPoints(corners, K, dist, P=new_K) if len(corners) else corners
    return out, np.asarray(moved, np.float64).reshape(-1, 2), np.asarray(new_K, np.float64)
