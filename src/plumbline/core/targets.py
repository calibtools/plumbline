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
CIRCLES = "circles"
ASYMMETRIC_CIRCLES = "asymmetric_circles"
APRILGRID = "aprilgrid"
KINDS = (CHARUCO, CHECKERBOARD, CIRCLES, ASYMMETRIC_CIRCLES, APRILGRID)
APRILGRID_BORDER_BITS = 2  # Kalibr prints its tags with a black border two cells wide
CIRCLE_KINDS = (CIRCLES, ASYMMETRIC_CIRCLES)
DEFAULT_CIRCLE_RATIO = 0.56  # circle diameter over spacing when the diameter is not given

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

    For circle grids squares_x / squares_y count the positions of a square
    lattice, square_mm is the lattice spacing and marker_mm the circle
    diameter (only drawing needs it). A symmetric grid has a circle at every
    position. An asymmetric grid has one at every other position,
    checkerboard-wise: a lattice of R rows x C columns holds R rows of C / 2
    circles, which OpenCV calls pattern size (C / 2, R).

    For an AprilGrid (Kalibr layout) squares_x / squares_y count tags,
    square_mm is the tag size including its black border and marker_mm the
    gap between neighbouring tags. The first ID is the bottom-left tag; IDs
    run left to right and then upward, and every tag is printed turned by
    180 degrees, as Kalibr does. The calibration points are the four outer
    corners of every tag.
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
        if self.kind not in KINDS:
            raise ValueError(f"unknown target kind: {self.kind}")
        if self.kind in CIRCLE_KINDS:
            if self.squares_x < 3 or self.squares_y < 3:
                raise ValueError("a circle grid needs at least 3 x 3 positions")
            if self.square_mm <= 0:
                raise ValueError("spacing must be positive")
            if self.kind == ASYMMETRIC_CIRCLES and (self.squares_x % 2 or self.squares_x < 4):
                raise ValueError("an asymmetric circle grid needs an even number of columns, at least 4, "
                                 "so that every row has the same number of circles")
            if self.marker_mm is not None and not 0 < self.marker_mm < self.square_mm:
                raise ValueError("circle diameter must be positive and smaller than the spacing")
            return
        if self.kind == APRILGRID:
            if self.squares_x < 2 or self.squares_y < 2:
                raise ValueError("an AprilGrid needs at least 2 x 2 tags")
            if self.square_mm <= 0:
                raise ValueError("tag size must be positive")
            if self.marker_mm is None or self.marker_mm <= 0:
                raise ValueError("the gap between tags must be positive")
            if self.dictionary not in DICTIONARIES or "APRILTAG" not in self.dictionary:
                raise ValueError("an AprilGrid needs an AprilTag family, for example DICT_APRILTAG_36h11")
            available = int(self.aruco_dictionary().bytesList.shape[0])
            last = self.first_marker_id + self.marker_count - 1
            if self.first_marker_id < 0 or last >= available:
                raise ValueError(
                    f"this board needs tag IDs {self.first_marker_id} to {last}, "
                    f"but {self.dictionary} only has IDs 0 to {available - 1}"
                )
            return
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
    def is_circles(self) -> bool:
        return self.kind in CIRCLE_KINDS

    @property
    def opencv_size(self) -> tuple[int, int]:
        """The size OpenCV's detector takes: (points per row, rows)."""
        if self.kind == CIRCLES:
            return (self.squares_x, self.squares_y)
        if self.kind == ASYMMETRIC_CIRCLES:
            return (self.squares_x // 2, self.squares_y)
        return (self.inner_cols, self.inner_rows)

    @property
    def corner_count(self) -> int:
        if self.kind == APRILGRID:
            return 4 * self.marker_count
        cols, rows = self.opencv_size
        return cols * rows

    @property
    def tag_step_mm(self) -> float:
        """Distance from one tag of an AprilGrid to the next: tag size plus gap."""
        return self.square_mm + float(self.marker_mm)

    def tag_origin(self, index: int) -> tuple[float, float]:
        """Top-left corner of the tag with the given position in ID order (0 = first ID)."""
        row, col = divmod(index, self.squares_x)  # row 0 is the bottom row
        return (col * self.tag_step_mm, (self.squares_y - 1 - row) * self.tag_step_mm)

    @property
    def circle_diameter_mm(self) -> float:
        if not self.is_circles:
            raise ValueError("only circle grids have a circle diameter")
        return float(self.marker_mm) if self.marker_mm else DEFAULT_CIRCLE_RATIO * self.square_mm

    @property
    def point_word(self) -> str:
        """What one detected point is called for this pattern."""
        return "circles" if self.is_circles else "corners"

    @property
    def marker_count(self) -> int:
        """Markers on a ChArUco board: one in every white square."""
        if self.kind == APRILGRID:
            return self.squares_x * self.squares_y
        if self.kind != CHARUCO:
            return 0
        return (self.squares_x * self.squares_y) // 2

    @property
    def width_mm(self) -> float:
        """Width of the printed pattern, without any margin."""
        if self.is_circles:
            return (self.squares_x - 1) * self.square_mm + self.circle_diameter_mm
        if self.kind == APRILGRID:
            return (self.squares_x - 1) * self.tag_step_mm + self.square_mm
        return self.squares_x * self.square_mm

    @property
    def height_mm(self) -> float:
        if self.is_circles:
            return (self.squares_y - 1) * self.square_mm + self.circle_diameter_mm
        if self.kind == APRILGRID:
            return (self.squares_y - 1) * self.tag_step_mm + self.square_mm
        return self.squares_y * self.square_mm

    @property
    def centre_mm(self) -> tuple[float, float]:
        """Middle of the pattern in the coordinates object_points() uses."""
        if self.is_circles:
            return ((self.squares_x - 1) * self.square_mm / 2, (self.squares_y - 1) * self.square_mm / 2)
        return (self.width_mm / 2, self.height_mm / 2)

    def object_points(self) -> np.ndarray:
        if self.is_circles:
            return self._circle_points()
        if self.kind == APRILGRID:
            return self._tag_points()
        return self._corner_points()

    def _tag_points(self) -> np.ndarray:
        """(4 N, 3) tag corners. Point 4 t + k is corner k of the tag with ID first_marker_id + t.

        The origin is the top-left corner of the block of tags, x to the right,
        y down. Corner k follows the order OpenCV's marker detector returns for
        a tag printed the Kalibr way: bottom-right, bottom-left, top-left, top-right.
        """
        s = self.square_mm
        pts = np.zeros((self.corner_count, 3), np.float32)
        for t in range(self.marker_count):
            x, y = self.tag_origin(t)
            pts[4 * t: 4 * t + 4, :2] = ((x + s, y + s), (x, y + s), (x, y), (x + s, y))
        return pts

    def _circle_points(self) -> np.ndarray:
        """(N, 3) circle centres in the order OpenCV's findCirclesGrid returns them.

        The origin is the centre of the first circle. In an asymmetric grid
        odd rows are shifted by one lattice step.
        """
        cols, rows = self.opencv_size
        pts = np.zeros((rows * cols, 3), np.float32)
        for i in range(rows):
            for j in range(cols):
                if self.kind == ASYMMETRIC_CIRCLES:
                    x = (2 * j + i % 2) * self.square_mm
                else:
                    x = j * self.square_mm
                pts[i * cols + j] = (x, i * self.square_mm, 0)
        return pts

    def _corner_points(self) -> np.ndarray:
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
        if self.kind not in (CHARUCO, APRILGRID):
            raise ValueError("only ChArUco and AprilGrid targets have a dictionary")
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

    @property
    def short_label(self) -> str:
        """A few words for tight places: the part number if it has one, else the pattern and its size."""
        if self.part_number:
            return self.part_number
        if self.kind == CHARUCO:
            return f"ChArUco {self.squares_x} x {self.squares_y}"
        if self.kind == APRILGRID:
            return f"AprilGrid {self.squares_x} x {self.squares_y}"
        if self.kind == CIRCLES:
            return f"Circles {self.squares_x} x {self.squares_y}"
        if self.kind == ASYMMETRIC_CIRCLES:
            return f"Asym. circles {self.squares_x // 2} x {self.squares_y}"
        return f"Checkerboard {self.inner_cols} x {self.inner_rows}"

    def describe(self) -> str:
        if self.part_number:
            return f"{self.part_number}: {self._describe_pattern()}"
        return self._describe_pattern()

    def _describe_pattern(self) -> str:
        if self.kind == CHARUCO:
            text = (
                f"ChArUco {self.squares_x} x {self.squares_y}, "
                f"{self.square_mm:g} / {self.marker_mm:g} mm, {self.dictionary}"
            )
            if self.first_marker_id:
                text += f", IDs from {self.first_marker_id}"
            return text
        if self.kind == APRILGRID:
            text = (f"AprilGrid {self.squares_x} x {self.squares_y} tags, "
                    f"{self.square_mm:g} mm tags, {self.marker_mm:g} mm gap, {self.dictionary}")
            if self.first_marker_id:
                text += f", IDs from {self.first_marker_id}"
            return text
        if self.kind == CIRCLES:
            return f"Circle grid {self.squares_x} x {self.squares_y}, {self.square_mm:g} mm spacing"
        if self.kind == ASYMMETRIC_CIRCLES:
            return (f"Asymmetric circle grid, {self.squares_y} rows of {self.squares_x // 2}, "
                    f"{self.square_mm:g} mm lattice spacing")
        return (
            f"Checkerboard {self.inner_cols} x {self.inner_rows} inner corners, "
            f"{self.square_mm:g} mm"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Target":
        return Target(**data)
