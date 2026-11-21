"""Target detection in a single image."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .targets import ASYMMETRIC_CIRCLES, CHARUCO, Target

# A view needs enough corners to constrain its pose well.
MIN_CORNERS = 8


@dataclass
class Detection:
    """Corners found in one image.

    ids index into Target.object_points(). corners are (n, 2) pixel
    positions with the origin at the centre of the top-left pixel.
    """

    path: str
    width: int
    height: int
    ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    note: str = ""

    @property
    def count(self) -> int:
        return int(len(self.ids))

    @property
    def usable(self) -> bool:
        return self.count >= MIN_CORNERS

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "ids": self.ids.astype(int).tolist(),
            "corners": np.round(self.corners.astype(float), 4).tolist(),
            "note": self.note,
        }

    @staticmethod
    def from_dict(data: dict) -> "Detection":
        return Detection(
            path=data["path"],
            width=int(data["width"]),
            height=int(data["height"]),
            ids=np.asarray(data["ids"], np.int32).reshape(-1),
            corners=np.asarray(data["corners"], np.float32).reshape(-1, 2),
            note=data.get("note", ""),
        )


def read_gray(path: str) -> np.ndarray | None:
    """Read an image as 8-bit grey. Returns None if it cannot be read."""
    data = np.fromfile(path, dtype=np.uint8)  # handles non-ASCII paths on Windows
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def detect(gray: np.ndarray, target: Target, path: str = "") -> Detection:
    """Find target corners in a grey image."""
    h, w = gray.shape[:2]
    det = Detection(path=path, width=w, height=h)
    if target.kind == CHARUCO:
        _detect_charuco(gray, target, det)
    elif target.is_circles:
        _detect_circles(gray, target, det)
    else:
        _detect_checkerboard(gray, target, det)
    if det.count == 0:
        det.note = det.note or "No target found"
    elif not det.usable:
        det.note = f"Only {det.count} {target.point_word} found, at least {MIN_CORNERS} are needed"
    return det


def detect_file(path: str, target: Target) -> Detection:
    gray = read_gray(path)
    if gray is None:
        return Detection(path=path, width=0, height=0, note="Could not read this file")
    return detect(gray, target, path)


def _detect_charuco(gray: np.ndarray, target: Target, det: Detection) -> None:
    detector = cv2.aruco.CharucoDetector(target.charuco_board())
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if ids is None or len(ids) == 0:
        if marker_ids is not None and len(marker_ids) > 0:
            seen = np.asarray(marker_ids).reshape(-1)
            lo, hi = int(seen.min()), int(seen.max())
            first = target.first_marker_id
            last = first + target.marker_count - 1
            if hi < first or lo > last:
                det.note = (
                    f"The markers in this image have IDs {lo} to {hi}, but the target is set to "
                    f"IDs {first} to {last}. This looks like another board of a set: "
                    f"try First marker ID {lo - lo % target.marker_count}."
                )
            else:
                det.note = (
                    f"{len(seen)} markers found but no board corners. "
                    "Check the square count, the sizes and the pattern layout."
                )
        return
    det.ids = np.asarray(ids, np.int32).reshape(-1)
    det.corners = np.asarray(corners, np.float32).reshape(-1, 2)


def _detect_checkerboard(gray: np.ndarray, target: Target, det: Detection) -> None:
    size = (target.inner_cols, target.inner_rows)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
    if not found or corners is None:
        return
    corners = np.asarray(corners, np.float32).reshape(-1, 2)
    if len(corners) != target.corner_count:
        return
    det.ids = np.arange(target.corner_count, dtype=np.int32)
    det.corners = corners


def _detect_circles(gray: np.ndarray, target: Target, det: Detection) -> None:
    """Dark circles on a light board. The whole grid has to be in view."""
    h, w = gray.shape[:2]
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 12.0
    params.maxArea = max(5000.0, w * h / 40.0)  # the default upper limit rejects circles in large images
    params.minDistBetweenBlobs = 2.0
    blobs = cv2.SimpleBlobDetector_create(params)
    flag = cv2.CALIB_CB_ASYMMETRIC_GRID if target.kind == ASYMMETRIC_CIRCLES else cv2.CALIB_CB_SYMMETRIC_GRID
    found, centres = cv2.findCirclesGrid(gray, target.opencv_size, flags=flag, blobDetector=blobs)
    if not found or centres is None:
        seen = len(blobs.detect(gray))
        if seen:
            det.note = (f"{seen} dark spots found but no complete grid of {target.corner_count} circles. "
                        "The whole grid must be in the image; check the row and column counts.")
        return
    centres = np.asarray(centres, np.float32).reshape(-1, 2)
    if len(centres) != target.corner_count:
        return
    det.ids = np.arange(target.corner_count, dtype=np.int32)
    det.corners = centres
