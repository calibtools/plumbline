#!/usr/bin/env python3
"""Render synthetic ChArUco images using inverse mapping.

For each output pixel (u,v):
  1. Undistort (u,v) → normalized ray (x',y')
  2. Back-project through K⁻¹ to get 3D ray direction
  3. Intersect ray with the board plane (transformed by R,t)
  4. Map intersection to board-image texture coordinates
  5. Sample the board image

This correctly handles lens distortion (unlike warp-perspective which
only distorts the 4 corners and uses a planar homography for the interior).
"""
import json
import math
import os
import sys

import cv2
import numpy as np

# ---- Board config (same for both datasets) ----
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 30.0
MARKER_LENGTH = 22.0
BOARD_W_MM = SQUARES_X * SQUARE_LENGTH   # 150
BOARD_H_MM = SQUARES_Y * SQUARE_LENGTH   # 210
BOARD_PX_PER_MM = 10

# ---- Camera config ----
FX, FY = 800.0, 800.0
CX, CY = 640.0, 480.0
IMAGE_W, IMAGE_H = 1280, 960
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST = np.array([-0.1, 0.01, 0.001, -0.001, 0.0001], dtype=np.float64)

# ---- Noise ----
INTENSITY_NOISE_SIGMA = 3
BLUR_KSIZE = 3


def create_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    except:
        board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    return board, dictionary


def generate_board_image(board):
    """Render flat board image with border, in pixels."""
    bw = int(BOARD_W_MM * BOARD_PX_PER_MM)
    bh = int(BOARD_H_MM * BOARD_PX_PER_MM)
    border_px = int(SQUARE_LENGTH * BOARD_PX_PER_MM)
    tw = bw + 2 * border_px
    th = bh + 2 * border_px
    try:
        img = board.generateImage((tw, th), marginSize=border_px)
    except:
        img = board.draw((tw, th), marginSize=border_px)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img, border_px


def render_image_inverse(board_img, board_border_px, rvec, tvec, centered, rng):
    """Render a camera image by inverse-mapping each pixel through the lens model.

    Args:
        board_img: the flat board texture (with border)
        board_border_px: border width in board-image pixels
        rvec, tvec: board pose (if centered, board center is at origin)
        centered: whether board coords are centered at origin
        rng: numpy RandomState for noise
    """
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    # Board plane in world coords: z=0 (in board frame)
    # A point on the board (x_b, y_b, 0) maps to camera frame as R @ [x_b, y_b, 0]^T + t

    # Precompute: for each pixel (u,v), undistort to get normalized coords,
    # then intersect with the board plane.

    # Generate all pixel coordinates
    u_coords = np.arange(IMAGE_W, dtype=np.float64)
    v_coords = np.arange(IMAGE_H, dtype=np.float64)
    uu, vv = np.meshgrid(u_coords, v_coords)  # (H, W)

    # Stack into (N, 1, 2) for undistortPoints
    pts = np.stack([uu.ravel(), vv.ravel()], axis=-1).reshape(-1, 1, 2)

    # Undistort: pixel coords → normalized camera coords
    pts_undist = cv2.undistortPoints(pts, K, DIST)  # (N, 1, 2) normalized
    pts_norm = pts_undist.reshape(-1, 2)  # (N, 2)

    # Ray direction in camera frame: (x_n, y_n, 1)
    rays = np.hstack([pts_norm, np.ones((len(pts_norm), 1))])  # (N, 3)

    # Transform ray to board frame: R^T @ (ray * lambda - t) = board_point
    # Board plane is z=0 in board frame.
    # Camera point: P_cam = lambda * ray_cam
    # Board point: P_board = R^T @ (P_cam - t)
    # P_board.z = 0 → solve for lambda:
    #   R^T[2,:] @ (lambda * ray_cam - t) = 0
    #   lambda * (R^T[2,:] @ ray_cam) = R^T[2,:] @ t
    #   lambda = (R^T[2,:] @ t) / (R^T[2,:] @ ray_cam)

    Rt = R.T  # (3, 3)
    r2 = Rt[2, :]  # third row of R^T

    denom = rays @ r2  # (N,) = R^T[2,:] @ ray for each pixel
    numer = r2 @ t     # scalar = R^T[2,:] @ t

    # Avoid division by zero (rays parallel to board)
    valid = np.abs(denom) > 1e-10
    lam = np.full(len(rays), np.nan)
    lam[valid] = numer / denom[valid]

    # Board-frame coordinates
    P_cam = rays * lam[:, np.newaxis]  # (N, 3) camera-frame 3D points
    P_board = (Rt @ (P_cam - t).T).T  # (N, 3) board-frame coords

    # Board-image texture coordinates
    # Board image origin: the board pattern starts at (border_px, border_px).
    # Board coords: if centered, x_b ∈ [-W/2, W/2], y_b ∈ [-H/2, H/2]
    #               if not centered, x_b ∈ [0, W], y_b ∈ [0, H]
    # But the chessboard corners start at (sq_len, sq_len), and the board image
    # includes a border. The board image covers:
    #   x_mm ∈ [-border_mm, W+border_mm], y_mm ∈ [-border_mm, H+border_mm]  (non-centered)
    #   x_mm ∈ [-W/2-border_mm, W/2+border_mm] (centered)

    border_mm = board_border_px / BOARD_PX_PER_MM

    if centered:
        # Board coords centered: origin at board center
        x_mm = P_board[:, 0] + BOARD_W_MM / 2.0  # shift to [0, BOARD_W_MM]
        y_mm = P_board[:, 1] + BOARD_H_MM / 2.0
    else:
        x_mm = P_board[:, 0]
        y_mm = P_board[:, 1]

    # Map mm → board image pixels
    tex_x = (x_mm + border_mm) * BOARD_PX_PER_MM
    tex_y = (y_mm + border_mm) * BOARD_PX_PER_MM

    # Sample from board image
    bh, bw = board_img.shape[:2]
    tex_x_i = tex_x.astype(np.float32).reshape(IMAGE_H, IMAGE_W)
    tex_y_i = tex_y.astype(np.float32).reshape(IMAGE_H, IMAGE_W)

    # Use cv2.remap for bilinear interpolation
    canvas = cv2.remap(board_img, tex_x_i, tex_y_i, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(200, 200, 200))

    # Mark invalid pixels (behind camera or not on board) as background
    lam_img = lam.reshape(IMAGE_H, IMAGE_W)
    invalid = (lam_img <= 0) | np.isnan(lam_img)
    # Also mark pixels outside the board+border area
    x_mm_img = x_mm.reshape(IMAGE_H, IMAGE_W)
    y_mm_img = y_mm.reshape(IMAGE_H, IMAGE_W)
    outside = ((x_mm_img < -border_mm) | (x_mm_img > BOARD_W_MM + border_mm) |
               (y_mm_img < -border_mm) | (y_mm_img > BOARD_H_MM + border_mm))
    mask = invalid | outside
    canvas[mask] = 200  # grey background

    # Add noise and blur
    noise = rng.normal(0, INTENSITY_NOISE_SIGMA, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    canvas = cv2.GaussianBlur(canvas, (BLUR_KSIZE, BLUR_KSIZE), 0)

    return canvas


def random_pose_good(zone_row, zone_col, area_frac, tilt_deg, rng):
    """Compute a centered pose for the good dataset."""
    zone_w = IMAGE_W / 3.0
    zone_h = IMAGE_H / 3.0
    target_px = zone_w * (zone_col + 0.5) + rng.uniform(-zone_w * 0.3, zone_w * 0.3)
    target_py = zone_h * (zone_row + 0.5) + rng.uniform(-zone_h * 0.3, zone_h * 0.3)
    target_px = np.clip(target_px, 80, IMAGE_W - 80)
    target_py = np.clip(target_py, 80, IMAGE_H - 80)

    # Depth from area fraction (compensate for tilt foreshortening)
    cos_tilt = max(0.3, math.cos(tilt_deg * math.pi / 180))
    adj_area = area_frac / cos_tilt
    board_area = BOARD_W_MM * BOARD_H_MM
    tz = math.sqrt(board_area * FX * FY / (adj_area * IMAGE_W * IMAGE_H))

    tx = (target_px - CX) * tz / FX
    ty = (target_py - CY) * tz / FY

    # Rotation
    azimuth = rng.uniform(0, 2 * math.pi)
    axis = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    tilt_rad = tilt_deg * math.pi / 180.0
    z_angle = rng.uniform(-15, 15) * math.pi / 180.0
    R_tilt, _ = cv2.Rodrigues((axis * tilt_rad).reshape(3, 1))
    R_z, _ = cv2.Rodrigues(np.array([0, 0, z_angle]).reshape(3, 1))
    R = R_tilt @ R_z
    rvec, _ = cv2.Rodrigues(R)

    return rvec, np.array([[tx], [ty], [tz]], dtype=np.float64)


def random_pose_poor(rng):
    """Original poor-dataset pose: small tx/ty, board near center."""
    angle = rng.uniform(5, 45) * np.pi / 180
    axis = rng.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    rvec = (axis * angle).reshape(3, 1)
    tx = rng.uniform(-80, 80)
    ty = rng.uniform(-60, 60)
    tz = rng.uniform(400, 800)
    return rvec.astype(np.float64), np.array([[tx], [ty], [tz]], dtype=np.float64)


def generate_dataset(out_dir, mode, seed):
    """Generate a full synthetic dataset.

    mode: 'good' or 'poor'
    """
    rng = np.random.RandomState(seed)
    board, _ = create_board()
    board_img, border_px = generate_board_image(board)
    os.makedirs(out_dir, exist_ok=True)

    centered = (mode == 'good')
    n_images = 25

    if mode == 'good':
        # Zone assignments: 9 zones × ≥2 each + 7 random
        zones = []
        for r in range(3):
            for c in range(3):
                zones.extend([(r, c)] * 2)
        for _ in range(n_images - 18):
            zones.append((rng.randint(0, 3), rng.randint(0, 3)))
        rng.shuffle(zones)
        zones = zones[:n_images]

        tilts = np.linspace(0, 45, n_images)
        rng.shuffle(tilts)
        areas = np.linspace(0.16, 0.34, n_images)
        rng.shuffle(areas)

    per_image = []
    for i in range(n_images):
        if mode == 'good':
            rvec, tvec = random_pose_good(zones[i][0], zones[i][1], areas[i], tilts[i], rng)
        else:
            rvec, tvec = random_pose_poor(rng)

        img = render_image_inverse(board_img, border_px, rvec, tvec, centered, rng)
        fname = f'img_{i:02d}.png'
        cv2.imwrite(os.path.join(out_dir, fname), img)

        per_image.append({
            'filename': fname,
            'rvec': rvec.flatten().tolist(),
            'tvec': tvec.flatten().tolist(),
        })
        print(f'  [{i+1:2d}/{n_images}] {fname}')

    truth = {
        'camera_matrix': K.flatten().tolist(),
        'dist_coeffs': DIST.tolist(),
        'image_size': [IMAGE_W, IMAGE_H],
        'board': {
            'squares_x': SQUARES_X, 'squares_y': SQUARES_Y,
            'square_length': SQUARE_LENGTH, 'marker_length': MARKER_LENGTH,
            'dictionary': 'DICT_4X4_50',
        },
        'noise': {
            'corner_sigma_px': 0,
            'intensity_sigma': INTENSITY_NOISE_SIGMA,
            'blur_ksize': BLUR_KSIZE,
        },
        'images': per_image,
    }
    truth_path = os.path.join(out_dir, 'truth.json')
    with open(truth_path, 'w') as f:
        json.dump(truth, f, indent=2)
    print(f'  Wrote {truth_path}')


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    print('=== Generating synthetic-01 (good coverage, inverse mapping) ===')
    generate_dataset(os.path.join(project_root, 'fixtures', 'synthetic-01'), 'good', seed=123)

    print('\n=== Generating synthetic-poor-01 (poor coverage, inverse mapping) ===')
    generate_dataset(os.path.join(project_root, 'fixtures', 'synthetic-poor-01'), 'poor', seed=42)

    print('\nDone. Run calibrate_reference.py on each to generate reference.json.')
