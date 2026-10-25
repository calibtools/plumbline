#!/usr/bin/env python3
"""Render 5 synthetic ChArUco board images at 4000x3000 resolution.

These are for timing/benchmarking only -- no truth.json or reference.json
is produced.  Same board parameters as render_synthetic.py (5x7, sq=30,
mk=22, DICT_4X4_50) with camera intrinsics scaled for the larger image.
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Board configuration (same as render_synthetic.py)
# ---------------------------------------------------------------------------
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 30.0   # mm
MARKER_LENGTH = 22.0   # mm

# ---------------------------------------------------------------------------
# Camera parameters scaled for 4000x3000
# ---------------------------------------------------------------------------
FX, FY = 2400.0, 2400.0
CX, CY = 2000.0, 1500.0
IMAGE_W, IMAGE_H = 4000, 3000

CAMERA_MATRIX = np.array([
    [FX,  0, CX],
    [ 0, FY, CY],
    [ 0,  0,  1],
], dtype=np.float64)

DIST_COEFFS = np.array([-0.1, 0.01, 0.001, -0.001, 0.0001], dtype=np.float64)

NUM_IMAGES = 5
SEED = 99

INTENSITY_NOISE_SIGMA = 3
BLUR_KSIZE = 3
BOARD_PX_PER_MM = 10


def _create_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH,
            MARKER_LENGTH,
            dictionary,
        )
    except Exception:
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X, SQUARES_Y,
            SQUARE_LENGTH, MARKER_LENGTH,
            dictionary,
        )
    return board, dictionary


def _generate_board_image(board):
    board_w = int(SQUARES_X * SQUARE_LENGTH * BOARD_PX_PER_MM)
    board_h = int(SQUARES_Y * SQUARE_LENGTH * BOARD_PX_PER_MM)
    border = int(SQUARE_LENGTH * BOARD_PX_PER_MM)
    total_w = board_w + 2 * border
    total_h = board_h + 2 * border
    try:
        board_img = board.generateImage((total_w, total_h), marginSize=border)
    except Exception:
        board_img = board.draw((total_w, total_h), marginSize=border)
    if len(board_img.shape) == 2:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    return board_img, border


def _random_pose(rng):
    """Return (rvec, tvec) with gentle angles and good depth for detection."""
    angle = rng.uniform(5, 30) * np.pi / 180.0
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + 1e-9
    rvec = (axis * angle).reshape(3, 1)

    tx = rng.uniform(-50, 50)
    ty = rng.uniform(-40, 40)
    tz = rng.uniform(500, 900)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)
    return rvec, tvec


def _four_board_corners_3d(border_mm, board_w_mm, board_h_mm):
    return np.array([
        [-border_mm,              -border_mm,              0],
        [board_w_mm + border_mm,  -border_mm,              0],
        [board_w_mm + border_mm,  board_h_mm + border_mm,  0],
        [-border_mm,              board_h_mm + border_mm,  0],
    ], dtype=np.float64)


def _render_image(board_img, rvec, tvec, rng, border_px):
    board_w_mm = SQUARES_X * SQUARE_LENGTH
    board_h_mm = SQUARES_Y * SQUARE_LENGTH
    border_mm = border_px / BOARD_PX_PER_MM

    corners_3d = _four_board_corners_3d(border_mm, board_w_mm, board_h_mm)
    corners_2d, _ = cv2.projectPoints(
        corners_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS,
    )
    corners_2d = corners_2d.reshape(-1, 2).astype(np.float32)

    h, w = board_img.shape[:2]
    src_pts = np.array([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, corners_2d)
    canvas = np.full((IMAGE_H, IMAGE_W, 3), 200, dtype=np.uint8)
    cv2.warpPerspective(
        board_img, H, (IMAGE_W, IMAGE_H),
        dst=canvas,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    noise = rng.normal(0, INTENSITY_NOISE_SIGMA, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    canvas = cv2.GaussianBlur(canvas, (BLUR_KSIZE, BLUR_KSIZE), 0)
    return canvas


def main():
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    board, dictionary = _create_board()
    board_img, border_px = _generate_board_image(board)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_dir = os.path.join(project_root, "fixtures", "synthetic-large")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(NUM_IMAGES):
        rvec, tvec = _random_pose(rng)
        img = _render_image(board_img, rvec, tvec, rng, border_px)
        fname = f"img_{i:02d}.png"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        print(f"  [{i+1}/{NUM_IMAGES}] {fname}  ({IMAGE_W}x{IMAGE_H})")

    print(f"\nWrote {NUM_IMAGES} images to {out_dir}")


if __name__ == "__main__":
    main()
