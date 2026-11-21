"""Calibration target definitions.

One place defines the geometry of a target. Detection, calibration,
rendering and export all read it from here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

CHARUCO = "charuco"
CHECKERBOARD = "checkerboard"

DICTIONARIES = [
    name
    for name in (
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
    )
    if hasattr(cv2.aruco, name)
]


@dataclass(frozen=True)
class Target:
    """A planar target. Lengths are in millimetres.

    squares_x / squares_y count squares, not inner corners. A 9 x 6
    checkerboard in OpenCV's inner-corner convention is 10 x 7 squares here.
    """

    kind: str
    squares_x: int
    squares_y: int
    square_mm: float
    marker_mm: float | None = None
    dictionary: str | None = None
    legacy_pattern: bool = False
    part_number: str | None = None
    first_marker_id: int = 0  # boards in a set often continue the numbering of the board before

    def __post_init__(self) -> None:
        if self.kind not in (CHARUCO, CHECKERBOARD):
            raise ValueError(f"unknown target kind: {self.kind}")
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("a target needs at least 3 x 3 squares")
        if self.square_mm <= 0:
            raise ValueError("square size must be positive")
        if self.kind == CHARUCO:
            if self.dictionary not in DICTIONARIES:
                raise ValueError(f"unknown dictionary: {self.dictionary}")
            if not self.marker_mm or not 0 < self.marker_mm < self.square_mm:
                raise ValueError("marker size must be positive and smaller than the square")
            available = int(self.aruco_dictionary().bytesList.shape[0])
            last = self.first_marker_id + self.marker_count - 1
            if self.first_marker_id < 0 or last >= available:
                raise ValueError(
                    f"this board needs marker IDs {self.first_marker_id} to {last}, "
                    f"but {self.dictionary} only has IDs 0 to {available - 1}"
                )

    @property
    def inner_cols(self) -> int:
        return self.squares_x - 1

    @property
    def inner_rows(self) -> int:
        return self.squares_y - 1

    @property
    def corner_count(self) -> int:
        return self.inner_cols * self.inner_rows

    @property
    def marker_count(self) -> int:
        """Markers on a ChArUco board: one in every white square."""
        if self.kind != CHARUCO:
            return 0
        return (self.squares_x * self.squares_y) // 2

    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.square_mm

    def object_points(self) -> np.ndarray:
        """(N, 3) float32 inner-corner positions, row by row, z = 0.

        Corner id i sits at row i // inner_cols, column i % inner_cols.
        The origin is the outer corner of the first square, so the first
        inner corner is at (square, square, 0). This matches OpenCV's
        CharucoBoard.getChessboardCorners().
        """
        cols, rows = self.inner_cols, self.inner_rows
        pts = np.zeros((rows * cols, 3), np.float32)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        pts[:, :2] = (grid + 1) * self.square_mm
        return pts

    def aruco_dictionary(self):
        if self.kind != CHARUCO:
            raise ValueError("only ChArUco targets have a dictionary")
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))

    def charuco_board(self):
        ids = np.arange(self.first_marker_id, self.first_marker_id + self.marker_count, dtype=np.int32)
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            float(self.square_mm),
            float(self.marker_mm),
            self.aruco_dictionary(),
            ids,
        )
        if self.legacy_pattern:
            board.setLegacyPattern(True)
        return board

    def describe(self) -> str:
        if self.kind == CHARUCO:
            text = (
                f"ChArUco {self.squares_x} x {self.squares_y}, "
                f"{self.square_mm:g} / {self.marker_mm:g} mm, {self.dictionary}"
            )
            if self.first_marker_id:
                text += f", IDs from {self.first_marker_id}"
            return text
        return (
            f"Checkerboard {self.inner_cols} x {self.inner_rows} inner corners, "
            f"{self.square_mm:g} mm"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Target":
        return Target(**data)
