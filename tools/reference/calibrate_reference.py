#!/usr/bin/env python3
"""Run OpenCV ChArUco calibration on a fixture directory and save reference.json.

Usage:
    python calibrate_reference.py                          # defaults to fixtures/synthetic-01/
    python calibrate_reference.py fixtures/real-01/
    python calibrate_reference.py /absolute/path/to/images
"""

import glob
import json
import math
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Board configuration — must match render_synthetic.py and the browser code
# ---------------------------------------------------------------------------
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 30.0   # mm
MARKER_LENGTH = 22.0   # mm


def _create_board():
    """Return (board, dictionary) using the 4.7+ API with legacy fallback."""
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


def _create_detector(dictionary):
    """Return a CharucoDetector (4.7+) or None (use legacy path)."""
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.CharucoDetector(
            _create_board()[0],
            cv2.aruco.CharucoParameters(),
            params,
        )
        return detector
    except Exception:
        return None


def _detect_charuco(img, board, dictionary, detector):
    """Detect charuco corners; return (charuco_corners, charuco_ids, n_markers).

    Returns (None, None, 0) when detection fails.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if detector is not None:
        # 4.7+ unified detector API
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            detector.detectBoard(grey)
        )
        n_markers = 0 if marker_ids is None else len(marker_ids)
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
            return None, None, n_markers
        return charuco_corners, charuco_ids, n_markers

    # Legacy two-step path
    params = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(grey, dictionary, parameters=params)
    n_markers = 0 if marker_ids is None else len(marker_ids)
    if marker_ids is None or len(marker_ids) < 2:
        return None, None, n_markers

    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, grey, board,
    )
    if ret < 4:
        return None, None, n_markers
    return charuco_corners, charuco_ids, n_markers


def _per_image_rms(obj_points, img_points, rvec, tvec, camera_matrix, dist_coeffs):
    """Compute reprojection RMS for a single image."""
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    err = (img_points.reshape(-1, 2) - projected.reshape(-1, 2))
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    # Resolve fixture directory -------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    if len(sys.argv) > 1:
        fixture_dir = sys.argv[1]
        if not os.path.isabs(fixture_dir):
            fixture_dir = os.path.join(project_root, fixture_dir)
    else:
        fixture_dir = os.path.join(project_root, "fixtures", "synthetic-01")

    fixture_dir = os.path.abspath(fixture_dir)

    if not os.path.isdir(fixture_dir):
        print(f"Error: directory not found: {fixture_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect images ------------------------------------------------------
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(fixture_dir, ext)))
    image_paths.sort()

    if not image_paths:
        print(f"Error: no images found in {fixture_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {fixture_dir}")

    # Board & detector ----------------------------------------------------
    board, dictionary = _create_board()
    detector = _create_detector(dictionary)

    # Detect corners in all images ----------------------------------------
    all_charuco_corners = []
    all_charuco_ids = []
    per_image_info = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  SKIP (unreadable): {os.path.basename(path)}")
            continue

        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif (w, h) != image_size:
            print(f"  SKIP (size mismatch): {os.path.basename(path)}")
            continue

        corners, ids, n_markers = _detect_charuco(img, board, dictionary, detector)
        fname = os.path.basename(path)

        if corners is None:
            print(f"  SKIP (no charuco): {fname}  markers={n_markers}")
            per_image_info.append({
                "filename": fname,
                "charuco_ids": [],
                "charuco_corners": [],
                "num_markers": n_markers,
                "used": False,
            })
            continue

        all_charuco_corners.append(corners)
        all_charuco_ids.append(ids)

        per_image_info.append({
            "filename": fname,
            "charuco_ids": ids.flatten().tolist(),
            "charuco_corners": corners.reshape(-1, 2).tolist(),
            "num_markers": n_markers,
            "used": True,
        })
        print(f"  OK   {fname}  corners={len(ids)}  markers={n_markers}")

    used_count = sum(1 for x in per_image_info if x.get("used"))
    if used_count < 3:
        print(f"Error: only {used_count} usable images (need >= 3)", file=sys.stderr)
        sys.exit(1)

    print(f"\nCalibrating with {used_count} images ...")

    # Calibrate -----------------------------------------------------------
    # Build object/image point arrays for calibrateCamera
    try:
        obj_points_all = []
        img_points_all = []
        try:
            board_obj_pts = board.getChessboardCorners()
        except Exception:
            board_obj_pts = board.chessboardCorners
        board_obj_pts = board_obj_pts.reshape(-1, 3).astype(np.float32)

        for corners, ids in zip(all_charuco_corners, all_charuco_ids):
            obj_pts = board_obj_pts[ids.flatten()]
            obj_points_all.append(obj_pts.reshape(-1, 1, 3))
            img_points_all.append(corners.reshape(-1, 1, 2).astype(np.float32))

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_all,
            img_points_all,
            image_size,
            None,
            None,
        )
    except Exception as e:
        print(f"calibrateCamera failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"RMS reprojection error: {rms:.6f}")

    # Per-image RMS -------------------------------------------------------
    per_rms = []
    used_idx = 0
    for info in per_image_info:
        if not info.get("used"):
            per_rms.append(None)
            continue
        r = _per_image_rms(
            obj_points_all[used_idx],
            img_points_all[used_idx],
            rvecs[used_idx],
            tvecs[used_idx],
            camera_matrix,
            dist_coeffs,
        )
        per_rms.append(round(r, 6))
        used_idx += 1

    # Attach per-image RMS to info records
    for info, r in zip(per_image_info, per_rms):
        info["rms"] = r

    # Write reference.json ------------------------------------------------
    result = {
        "board": {
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length": SQUARE_LENGTH,
            "marker_length": MARKER_LENGTH,
            "dictionary": "DICT_4X4_50",
        },
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.flatten().tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "rms": round(rms, 6),
        "per_image_rms": [r for r in per_rms if r is not None],
        "images": per_image_info,
    }

    out_path = os.path.join(fixture_dir, "reference.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    # Summary -------------------------------------------------------------
    print(f"\nCamera matrix:\n{camera_matrix}")
    print(f"\nDistortion coefficients: {dist_coeffs.flatten()}")
    print(f"Image size: {image_size}")
    print(f"Overall RMS: {rms:.6f}")


if __name__ == "__main__":
    main()
