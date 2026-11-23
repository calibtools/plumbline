"""Several boards of the same kind in every image, each found by its own marker numbers.

The boards of a set continue the numbering: the second starts where the first ends. Each
board in each image is one view for the calibration, with a position of its own; nothing
assumes that the boards are fixed to each other. The flat list handed to `calibrate` holds
board 1's detections first, then board 2's, and so on: view `b * images + i` is board b + 1
in image i.

How the boards sit to each other is worked out afterwards, from every image that shows
two of them, together with how much that varies between images.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .detect import Detection
from .targets import APRILGRID, ARUCO_GRID, CHARUCO, Target

MOST_BOARDS = 4
NUMBERED = (CHARUCO, APRILGRID, ARUCO_GRID)     # patterns whose markers tell one board from another


def board_targets(target: Target, count: int) -> list[Target]:
    """Board 1 as entered, and the boards that follow it in the numbering. Raises ValueError if the dictionary runs out."""
    if count > 1 and target.kind not in NUMBERED:
        raise ValueError("Several boards in one image need a pattern with numbered markers: ChArUco, AprilGrid or ArUco grid.")
    return [target if b == 0 else replace(target, first_marker_id=target.first_marker_id + b * target.marker_count)
            for b in range(count)]


def together(per_board: list[list[Detection]]) -> list[Detection]:
    """The flat list for `calibrate`. Views of boards after the first carry the board in their name."""
    out = list(per_board[0])
    for b, dets in enumerate(per_board[1:], 2):
        out += [Detection(f"{d.path} [board {b}]", d.width, d.height, d.ids, d.corners, d.note) for d in dets]
    return out


def excluded_views(excluded: set[int], images: int, count: int) -> set[int]:
    """An image left out is left out with every board in it."""
    return {b * images + i for i in excluded for b in range(count)}


@dataclass
class ImageFit:
    """What the boards of one image add up to, for a table row, a bar and the drawing over the image."""
    index: int
    corner_count: int
    rms_point: float
    max_error: float
    residuals: np.ndarray       # (n, 2), board 1's points first
    corners: np.ndarray         # (n, 2), in the same order


def image_fit(result, per_board: list[list[Detection]], image: int) -> ImageFit | None:
    images = len(per_board[0])
    by_index = {v.index: v for v in result.views}
    res, pts = [], []
    for b, dets in enumerate(per_board):
        v = by_index.get(b * images + image)
        if v is not None:
            res.append(np.asarray(v.residuals, np.float64))
            pts.append(np.asarray(dets[image].corners, np.float64))
    if not res:
        return None
    res, pts = np.concatenate(res), np.concatenate(pts)
    sq = (res**2).sum(axis=1)
    return ImageFit(image, int(len(res)), float(np.sqrt(sq.mean())), float(np.sqrt(sq.max())), res, pts)


@dataclass
class BoardPlace:
    """One board after the first, seen from board 1."""
    board: int                       # counted from 0
    images: int                      # images that show this board and board 1
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))      # a point of this board, in board 1's frame:
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rotation @ p + translation, mm
    angle_spread_deg: float = float("nan")    # how much the images disagree, RMS about the place given
    place_spread_mm: float = float("nan")

    @property
    def angle_deg(self) -> float:
        return float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(self.rotation)[0])))

    @property
    def apart_mm(self) -> float:
        return float(np.linalg.norm(self.translation))

    def to_dict(self) -> dict:
        return {"board": self.board + 1, "images": self.images, "angle_deg": self.angle_deg, "apart_mm": self.apart_mm,
                "rotation": [[float(v) for v in row] for row in self.rotation],
                "translation_mm": [float(v) for v in self.translation],
                "angle_spread_deg": self.angle_spread_deg, "place_spread_mm": self.place_spread_mm}


def board_places(result, images: int, count: int) -> list[BoardPlace]:
    """Where boards 2, 3, ... sit relative to board 1, from the images that show both."""
    by_index = {v.index: v for v in result.views}

    def rot(v):
        return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]

    out = []
    for b in range(1, count):
        seen = []
        for i in range(images):
            first, other = by_index.get(i), by_index.get(b * images + i)
            if first is not None and other is not None:
                R1 = rot(first.rvec)
                seen.append((R1.T @ rot(other.rvec), R1.T @ (np.asarray(other.tvec, np.float64).ravel()
                                                             - np.asarray(first.tvec, np.float64).ravel())))
        if not seen:
            out.append(BoardPlace(b, 0))
            continue
        # The estimate closest to all the others stands for them: one image with a poor pose does not pull it.
        scale = max(float(np.median([np.linalg.norm(t) for _R, t in seen])), 1.0)
        cost = [sum(np.linalg.norm(cv2.Rodrigues(R @ R2.T)[0]) * scale + np.linalg.norm(t - t2) for R2, t2 in seen)
                for R, t in seen]
        R, t = seen[int(np.argmin(cost))]
        angles = [np.rad2deg(np.linalg.norm(cv2.Rodrigues(R2 @ R.T)[0])) for R2, _t in seen]
        places = [np.linalg.norm(t2 - t) for _R, t2 in seen]
        many = len(seen) > 1
        out.append(BoardPlace(b, len(seen), R, t,
                              float(np.sqrt(np.sum(np.square(angles)) / (len(seen) - 1))) if many else float("nan"),
                              float(np.sqrt(np.sum(np.square(places)) / (len(seen) - 1))) if many else float("nan")))
    return out
