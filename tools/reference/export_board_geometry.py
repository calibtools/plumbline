#!/usr/bin/env python3
"""Export ChArUco board chessboard corner geometry to JSON.

Creates two files:
  fixtures/board-geometry/standard.json  -- legacy=False (new corner order)
  fixtures/board-geometry/legacy.json    -- legacy=True  (old corner order)

Each file contains an array of [x, y, z] for corner IDs 0..23.
"""

import json
import os
import cv2
import numpy as np

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 30.0
MARKER_LENGTH = 22.0


def _make_board(legacy):
    """Create a CharucoBoard with the given legacy flag."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH,
            MARKER_LENGTH,
            dictionary,
        )
        # setLegacyPattern available in cv2 >= 4.7
        if hasattr(board, 'setLegacyPattern'):
            board.setLegacyPattern(legacy)
    except Exception:
        # Very old API -- no legacy flag support
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X, SQUARES_Y,
            SQUARE_LENGTH, MARKER_LENGTH,
            dictionary,
        )
    return board


def _get_corners(board):
    """Return chessboard corners as a list of [x, y, z] lists."""
    try:
        pts = board.getChessboardCorners()
    except Exception:
        pts = board.chessboardCorners
    pts = pts.reshape(-1, 3)
    return [[float(p[0]), float(p[1]), float(p[2])] for p in pts]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_dir = os.path.join(project_root, "fixtures", "board-geometry")
    os.makedirs(out_dir, exist_ok=True)

    # Standard (legacy=False)
    board_std = _make_board(legacy=False)
    corners_std = _get_corners(board_std)
    std_path = os.path.join(out_dir, "standard.json")
    with open(std_path, "w") as f:
        json.dump(corners_std, f, indent=2)
    print(f"Wrote {std_path}  ({len(corners_std)} corners)")

    # Legacy (legacy=True)
    board_leg = _make_board(legacy=True)
    corners_leg = _get_corners(board_leg)
    leg_path = os.path.join(out_dir, "legacy.json")
    with open(leg_path, "w") as f:
        json.dump(corners_leg, f, indent=2)
    print(f"Wrote {leg_path}  ({len(corners_leg)} corners)")


if __name__ == "__main__":
    main()
