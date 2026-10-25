#!/usr/bin/env python3
"""Render 25 synthetic ChArUco board images with known camera parameters.

The images are saved to fixtures/synthetic-01/ along with a truth.json that
records the exact camera matrix, distortion coefficients, and per-image poses.
"""

import json
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Board configuration
# ---------------------------------------------------------------------------
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 30.0   # mm
MARKER_LENGTH = 22.0   # mm

# ---------------------------------------------------------------------------
# Known (true) camera parameters
# ---------------------------------------------------------------------------
FX, FY = 800.0, 800.0
CX, CY = 640.0, 480.0
IMAGE_W, IMAGE_H = 1280, 960

CAMERA_MATRIX = np.array([
    [FX,  0, CX],
    [ 0, FY, CY],
    [ 0,  0,  1],
], dtype=np.float64)

DIST_COEFFS = np.array([-0.1, 0.01, 0.001, -0.001, 0.0001], dtype=np.float64)

NUM_IMAGES = 25
SEED = 42

# Noise parameters
CORNER_NOISE_SIGMA = 0.5   # pixels
INTENSITY_NOISE_SIGMA = 3  # grey levels
BLUR_KSIZE = 3

# Board image rendering resolution (pixels per mm) – higher gives sharper
# markers after warping.
BOARD_PX_PER_MM = 10


def _create_board():
    """Return a CharucoBoard using the 4.7+ API (fall back to legacy)."""
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
    """Render a flat board image with a white border."""
    board_w = int(SQUARES_X * SQUARE_LENGTH * BOARD_PX_PER_MM)
    board_h = int(SQUARES_Y * SQUARE_LENGTH * BOARD_PX_PER_MM)
    # Add a generous white border so corners near the edge are visible after
    # perspective warp.
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


def _board_corner_3d(board):
    """Return the 3-D positions (N x 3) of the *chessboard* (charuco) corners.

    These are the interior corners of the checkerboard pattern, in the board
    coordinate frame (Z=0 plane, origin at the top-left square corner, units
    in mm).
    """
    try:
        pts = board.getChessboardCorners()
    except Exception:
        pts = board.chessboardCorners
    return pts.reshape(-1, 3)


def _random_pose(rng):
    """Return (rvec, tvec) for a random but reasonable board pose."""
    # Rotation: random axis, angle up to 45 degrees
    angle = rng.uniform(5, 45) * np.pi / 180.0
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + 1e-9
    rvec = (axis * angle).reshape(3, 1)

    # Translation: board roughly centred in the image, at a reasonable depth
    tx = rng.uniform(-80, 80)
    ty = rng.uniform(-60, 60)
    tz = rng.uniform(400, 800)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

    return rvec, tvec


def _four_board_corners_3d(border_mm, board_w_mm, board_h_mm):
    """Return four 3-D points at the corners of the rendered board image
    (including the white border), in the board coordinate system (mm)."""
    return np.array([
        [-border_mm,              -border_mm,              0],
        [board_w_mm + border_mm,  -border_mm,              0],
        [board_w_mm + border_mm,  board_h_mm + border_mm,  0],
        [-border_mm,              board_h_mm + border_mm,  0],
    ], dtype=np.float64)


def _render_image(board_img, board, rvec, tvec, rng, border_px):
    """Warp the flat board image into the camera frame and add noise."""
    board_w_mm = SQUARES_X * SQUARE_LENGTH
    board_h_mm = SQUARES_Y * SQUARE_LENGTH
    border_mm = border_px / BOARD_PX_PER_MM

    # Four corners of the rendered board image in 3-D (mm)
    corners_3d = _four_board_corners_3d(border_mm, board_w_mm, board_h_mm)

    # Project them into the camera image
    corners_2d, _ = cv2.projectPoints(
        corners_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS,
    )
    corners_2d = corners_2d.reshape(-1, 2).astype(np.float32)

    # Corresponding pixel coords in the flat board image
    h, w = board_img.shape[:2]
    src_pts = np.array([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, corners_2d)
    canvas = np.full((IMAGE_H, IMAGE_W, 3), 200, dtype=np.uint8)  # grey bg
    cv2.warpPerspective(
        board_img, H, (IMAGE_W, IMAGE_H),
        dst=canvas,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    # Intensity noise
    noise = rng.normal(0, INTENSITY_NOISE_SIGMA, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Slight blur
    canvas = cv2.GaussianBlur(canvas, (BLUR_KSIZE, BLUR_KSIZE), 0)

    return canvas


def main():
    rng = np.random.RandomState(SEED)

    board, dictionary = _create_board()
    board_img, border_px = _generate_board_image(board)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_dir = os.path.join(project_root, "fixtures", "synthetic-01")
    os.makedirs(out_dir, exist_ok=True)

    per_image = []

    for i in range(NUM_IMAGES):
        rvec, tvec = _random_pose(rng)
        img = _render_image(board_img, board, rvec, tvec, rng, border_px)
        fname = f"img_{i:02d}.png"
        cv2.imwrite(os.path.join(out_dir, fname), img)

        per_image.append({
            "filename": fname,
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
        })
        print(f"  [{i+1:2d}/{NUM_IMAGES}] {fname}")

    truth = {
        "camera_matrix": CAMERA_MATRIX.flatten().tolist(),
        "dist_coeffs": DIST_COEFFS.tolist(),
        "image_size": [IMAGE_W, IMAGE_H],
        "board": {
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length": SQUARE_LENGTH,
            "marker_length": MARKER_LENGTH,
            "dictionary": "DICT_4X4_50",
        },
        "noise": {
            "corner_sigma_px": CORNER_NOISE_SIGMA,
            "intensity_sigma": INTENSITY_NOISE_SIGMA,
            "blur_ksize": BLUR_KSIZE,
        },
        "images": per_image,
    }

    truth_path = os.path.join(out_dir, "truth.json")
    with open(truth_path, "w") as f:
        json.dump(truth, f, indent=2)
    print(f"\nWrote {truth_path}")


if __name__ == "__main__":
    main()
