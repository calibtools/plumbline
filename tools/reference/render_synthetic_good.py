#!/usr/bin/env python3
"""Render 25 synthetic ChArUco board images with good spatial coverage.

The board center is placed using a 3x3 zone grid so that every region of
the image receives at least 2 images.  Board area fraction varies between
12% and 35%, and tilt angles are distributed uniformly from 0 to 45 deg.

Output: fixtures/synthetic-01/  (images + truth.json)
"""

import json
import math
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Board configuration  (same as render_synthetic.py)
# ---------------------------------------------------------------------------
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 30.0   # mm
MARKER_LENGTH = 22.0   # mm

# ---------------------------------------------------------------------------
# Known (true) camera parameters  (same as render_synthetic.py)
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
SEED = 123

# Noise parameters  (same as render_synthetic.py)
CORNER_NOISE_SIGMA = 0.5   # pixels
INTENSITY_NOISE_SIGMA = 3  # grey levels
BLUR_KSIZE = 3

# Board image rendering resolution (pixels per mm)
BOARD_PX_PER_MM = 10

BOARD_W_MM = SQUARES_X * SQUARE_LENGTH   # 150 mm
BOARD_H_MM = SQUARES_Y * SQUARE_LENGTH   # 210 mm


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


def _board_center_offset():
    """Offset so board geometric center is at origin."""
    return np.array([BOARD_W_MM / 2.0, BOARD_H_MM / 2.0, 0.0], dtype=np.float64)


def _four_board_corners_3d(border_mm):
    """Four corners of the rendered board image (with border), centered."""
    off = _board_center_offset()
    return np.array([
        [-border_mm,               -border_mm,               0],
        [BOARD_W_MM + border_mm,   -border_mm,               0],
        [BOARD_W_MM + border_mm,   BOARD_H_MM + border_mm,   0],
        [-border_mm,               BOARD_H_MM + border_mm,   0],
    ], dtype=np.float64) - off


def _zone_assignments(rng):
    """Return a list of 25 (zone_row, zone_col) assignments using the 3x3 grid.

    Each of the 9 zones gets at least 2 images; the remaining 7 are
    distributed randomly.
    """
    zones = []
    # 2 per zone = 18
    for r in range(3):
        for c in range(3):
            zones.append((r, c))
            zones.append((r, c))
    # 7 more random
    for _ in range(NUM_IMAGES - 18):
        zones.append((int(rng.randint(0, 3)), int(rng.randint(0, 3))))
    rng.shuffle(zones)
    return zones[:NUM_IMAGES]


def _target_positions(zones, rng):
    """For each zone assignment, pick a target pixel position that spreads
    well across the 8x6 heatmap grid.

    Strategy: within each 3x3 zone, pick sub-positions that map to different
    8x6 cells.  A 3x3 zone contains roughly 2.67 x 2 cells of the 8x6 grid.
    We spread the images assigned to each zone across different sub-regions.
    """
    zone_w = IMAGE_W / 3.0   # ~426.7
    zone_h = IMAGE_H / 3.0   # ~320.0
    cell_w = IMAGE_W / 8.0   # 160
    cell_h = IMAGE_H / 6.0   # 160

    # Track which 8x6 cells are covered
    covered = set()
    targets = []

    # Group images by zone
    zone_images = {}
    for i, (zr, zc) in enumerate(zones):
        zone_images.setdefault((zr, zc), []).append(i)

    result = [None] * len(zones)

    for (zr, zc), indices in zone_images.items():
        # Pixel bounds of this zone
        x_min = zone_w * zc
        x_max = zone_w * (zc + 1)
        y_min = zone_h * zr
        y_max = zone_h * (zr + 1)

        # Generate candidate positions spread across the zone
        for idx, img_i in enumerate(indices):
            # Spread within zone: divide into sub-positions
            n = len(indices)
            # Use a grid within the zone
            sub_cols = min(n, 3)
            sub_rows = max(1, (n + sub_cols - 1) // sub_cols)
            sr = idx // sub_cols
            sc = idx % sub_cols

            frac_x = (sc + 0.5) / sub_cols
            frac_y = (sr + 0.5) / sub_rows

            tx = x_min + frac_x * (x_max - x_min) + rng.uniform(-20, 20)
            ty = y_min + frac_y * (y_max - y_min) + rng.uniform(-20, 20)

            # Clamp
            margin = 50
            tx = np.clip(tx, margin, IMAGE_W - margin)
            ty = np.clip(ty, margin, IMAGE_H - margin)

            result[img_i] = (tx, ty)

    return result


def _compute_pose(target_px, target_py, area_fraction, tilt_deg, rng):
    """Compute (rvec, tvec) so the board center projects to (target_px, target_py)
    with the specified area fraction and tilt angle.

    Board points are centered at origin (centering fix applied).
    """

    # Compute depth from desired area fraction
    # Board real-world area = BOARD_W_MM * BOARD_H_MM
    # Projected area ~ (board_w_mm * fx / tz) * (board_h_mm * fy / tz) = board_area_mm2 * fx*fy / tz^2
    # Desired: projected_area / (IMAGE_W * IMAGE_H) = area_fraction
    # => tz = sqrt(board_area_mm2 * fx * fy / (area_fraction * IMAGE_W * IMAGE_H))
    board_area_mm2 = BOARD_W_MM * BOARD_H_MM
    tz = math.sqrt(board_area_mm2 * FX * FY / (area_fraction * IMAGE_W * IMAGE_H))

    # Compute tx, ty so board center projects to (target_px, target_py)
    # With centered board: board center is at origin, so projected center =
    #   (cx + tx * fx / tz, cy + ty * fy / tz)
    tx = (target_px - CX) * tz / FX
    ty = (target_py - CY) * tz / FY

    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

    # Rotation: tilt about a random axis in the XY plane
    tilt_rad = tilt_deg * np.pi / 180.0
    # Random azimuth for the tilt axis
    azimuth = rng.uniform(0, 2 * np.pi)
    axis = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    # Add small Z rotation for variety
    z_angle = rng.uniform(-15, 15) * np.pi / 180.0
    rvec_tilt = (axis * tilt_rad).reshape(3, 1)
    rvec_z = np.array([[0], [0], [z_angle]], dtype=np.float64)

    # Compose rotations
    R_tilt, _ = cv2.Rodrigues(rvec_tilt)
    R_z, _ = cv2.Rodrigues(rvec_z)
    R = R_tilt @ R_z
    rvec, _ = cv2.Rodrigues(R)

    return rvec, tvec


def _render_image(board_img, rvec, tvec, rng, border_px):
    """Warp the flat board image into the camera frame and add noise."""
    border_mm = border_px / BOARD_PX_PER_MM

    corners_3d = _four_board_corners_3d(border_mm)

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

    # Intensity noise
    noise = rng.normal(0, INTENSITY_NOISE_SIGMA, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Slight blur
    canvas = cv2.GaussianBlur(canvas, (BLUR_KSIZE, BLUR_KSIZE), 0)

    return canvas


def _compute_board_center_and_area(rvec, tvec):
    """Return (center_x, center_y) in pixels and area fraction.

    Board points are centered at origin, so board center = origin.
    """
    # Project board center (origin)
    center_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    cx, cy = center_2d.reshape(2)

    # Project board corners (no border) to compute area
    half_w = BOARD_W_MM / 2.0
    half_h = BOARD_H_MM / 2.0
    corners = np.array([
        [-half_w, -half_h, 0],
        [ half_w, -half_h, 0],
        [ half_w,  half_h, 0],
        [-half_w,  half_h, 0],
    ], dtype=np.float64)
    pts_2d, _ = cv2.projectPoints(corners, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    pts = pts_2d.reshape(-1, 2)

    # Shoelace formula for polygon area
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    area = abs(area) / 2.0

    img_area = IMAGE_W * IMAGE_H
    return float(cx), float(cy), area / img_area


def _compute_tilt_deg(rvec):
    """Return the tilt angle in degrees (angle between board normal and Z axis)."""
    R, _ = cv2.Rodrigues(rvec)
    board_normal = R[:, 2]  # Z column = board normal after rotation
    cos_angle = abs(board_normal[2])
    cos_angle = min(1.0, max(-1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _get_charuco_corners_2d(board, rvec, tvec):
    """Project all charuco (chessboard interior) corners into image pixels."""
    try:
        pts_3d = board.getChessboardCorners()
    except Exception:
        pts_3d = board.chessboardCorners
    pts_3d = pts_3d.reshape(-1, 3).astype(np.float64)
    # Apply centering offset
    off = _board_center_offset()
    pts_3d_centered = pts_3d - off
    pts_2d, _ = cv2.projectPoints(pts_3d_centered, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    return pts_2d.reshape(-1, 2)


def _print_heatmap(all_corners_2d):
    """Print an 8x6 heatmap grid of charuco corner counts and return zero-cell count."""
    cell_w = IMAGE_W / 8.0
    cell_h = IMAGE_H / 6.0
    grid = [[0] * 8 for _ in range(6)]
    for pts in all_corners_2d:
        for px, py in pts:
            col = int(px / cell_w)
            row = int(py / cell_h)
            if 0 <= col < 8 and 0 <= row < 6:
                grid[row][col] += 1

    print("\n8x6 heatmap grid (charuco corner counts):")
    print("    " + "  ".join(f"c{c}" for c in range(8)))
    zero_count = 0
    for r in range(6):
        row_str = f"r{r}  " + "  ".join(f"{grid[r][c]:2d}" for c in range(8))
        print(row_str)
        zero_count += sum(1 for c in range(8) if grid[r][c] == 0)
    print(f"\nZero cells: {zero_count} / 48  (requirement: at most 4)")
    return zero_count


def main():
    rng = np.random.RandomState(SEED)

    board, dictionary = _create_board()
    board_img, border_px = _generate_board_image(board)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_dir = os.path.join(project_root, "fixtures", "synthetic-01")
    os.makedirs(out_dir, exist_ok=True)

    zones = _zone_assignments(rng)
    targets = _target_positions(zones, rng)

    # Distribute tilt angles uniformly 0-45 across 25 images
    tilt_angles = np.linspace(0, 45, NUM_IMAGES)
    rng.shuffle(tilt_angles)

    # Distribute area fractions uniformly 16%-34% (pre-tilt; tilt compensation
    # in the loop adjusts for foreshortening so actual projected area lands
    # roughly in 12%-35%)
    area_fractions = np.linspace(0.16, 0.34, NUM_IMAGES)
    rng.shuffle(area_fractions)

    per_image = []
    all_corners_2d = []

    print(f"Generating {NUM_IMAGES} images with 3x3 zone coverage ...\n")
    print(f"{'Image':<12} {'Zone':>6} {'Center(x,y)':>16} {'Area%':>7} {'Tilt':>6}")
    print("-" * 52)

    for i in range(NUM_IMAGES):
        zone_row, zone_col = zones[i]
        area_frac = float(area_fractions[i])
        tilt_deg = float(tilt_angles[i])
        target_px, target_py = targets[i]

        # Compensate area fraction for tilt foreshortening: projected area
        # scales roughly as cos(tilt), so request more area for tilted boards.
        tilt_rad = tilt_deg * np.pi / 180.0
        cos_tilt = max(0.3, math.cos(tilt_rad))
        adjusted_area_frac = area_frac / cos_tilt

        rvec, tvec = _compute_pose(target_px, target_py, adjusted_area_frac, tilt_deg, rng)

        img = _render_image(board_img, rvec, tvec, rng, border_px)
        fname = f"img_{i:02d}.png"
        cv2.imwrite(os.path.join(out_dir, fname), img)

        # Compute actual projected center and area
        cx, cy, actual_area = _compute_board_center_and_area(rvec, tvec)
        actual_tilt = _compute_tilt_deg(rvec)

        # Collect projected charuco corners for heatmap
        corners_2d = _get_charuco_corners_2d(board, rvec, tvec)
        all_corners_2d.append(corners_2d)

        per_image.append({
            "filename": fname,
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
        })

        print(f"  {fname:<10} ({zone_row},{zone_col})  ({cx:6.1f},{cy:6.1f})  "
              f"{actual_area*100:5.1f}%  {actual_tilt:5.1f}°")

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

    zero_cells = _print_heatmap(all_corners_2d)
    if zero_cells > 4:
        print(f"\nWARNING: {zero_cells} zero cells exceeds the max of 4!")
    else:
        print(f"\nCoverage check PASSED.")


if __name__ == "__main__":
    main()
