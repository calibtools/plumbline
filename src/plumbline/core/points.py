"""The same board point across all images.

A point that is misplaced on the board itself (a dent, a lifted corner of a
sticker, a printing fault) is off in the same direction in every image, as
seen from the board. Averaged over the images, random noise shrinks and such
a point stands out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typing import TYPE_CHECKING

from . import lens
from .detect import Detection

if TYPE_CHECKING:
    from .calibrate import CalibrationResult

MIN_VIEWS = 6          # a point seen in fewer images is not judged
SCORE = 25.0           # how far the average must be from zero, in units of its own scatter (squared)
MIN_OFFSET_PX = 0.08   # and it must matter: at least this much in the image


@dataclass
class PointStats:
    ids: np.ndarray        # (M,) index into Target.object_points()
    count: np.ndarray      # (M,) images the point was used in
    rms_px: np.ndarray     # (M,) residual of this point over its images
    mean_px: np.ndarray    # (M, 2) average of observed minus predicted, in the image
    offset_mm: np.ndarray  # (M, 2) the same on the board: where the point seems to sit, minus where it should
    score: np.ndarray      # (M,) squared distance of offset_mm from zero in units of its own scatter

    def suspects(self) -> list[int]:
        """Positions (into ids) of points that look misplaced on the board, strongest first."""
        size = np.linalg.norm(self.mean_px, axis=1)
        hit = np.nonzero((self.count >= MIN_VIEWS) & (self.score > SCORE) & (size > MIN_OFFSET_PX))[0]
        return sorted(hit.tolist(), key=lambda i: -float(self.score[i]))


def point_statistics(result: "CalibrationResult", detections: list[Detection]) -> PointStats:
    K, dist, fisheye = result.camera_matrix, result.distortion, result.fisheye
    obj_all = result.target.object_points().astype(np.float64)
    per_point: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    step = 1e-3 * max(1.0, float(result.target.square_mm))
    for view in result.views:
        ids = np.asarray(detections[view.index].ids).reshape(-1)
        obj = obj_all[ids]
        seen_minus_predicted = -np.asarray(view.residuals, np.float64)
        # How the image of each point moves when the point moves on the board, by x and by y.
        slopes = []
        for axis in range(2):
            d = np.zeros(3)
            d[axis] = step
            slopes.append((lens.project(obj + d, view.rvec, view.tvec, K, dist, fisheye)
                           - lens.project(obj - d, view.rvec, view.tvec, K, dist, fisheye)) / (2 * step))
        J = np.stack(slopes, axis=2)                       # (N, 2, 2) px per mm
        on_board = np.linalg.solve(J, seen_minus_predicted[:, :, None])[:, :, 0]
        for k, pid in enumerate(ids):
            per_point.setdefault(int(pid), []).append((seen_minus_predicted[k], on_board[k]))

    ids = np.array(sorted(per_point), np.int32)
    count = np.zeros(len(ids), np.int32)
    rms = np.zeros(len(ids))
    mean_px = np.zeros((len(ids), 2))
    offset = np.zeros((len(ids), 2))
    score = np.zeros(len(ids))
    for i, pid in enumerate(ids):
        px = np.array([a for a, _b in per_point[int(pid)]])
        mm = np.array([b for _a, b in per_point[int(pid)]])
        n = len(px)
        count[i] = n
        rms[i] = float(np.sqrt(np.mean(np.sum(px**2, axis=1))))
        mean_px[i] = px.mean(axis=0)
        offset[i] = mm.mean(axis=0)
        if n >= 3:
            cov = np.cov(mm.T) / n                         # scatter of the average
            try:
                score[i] = float(offset[i] @ np.linalg.solve(cov, offset[i]))
            except np.linalg.LinAlgError:
                score[i] = 0.0
    return PointStats(ids, count, rms, mean_px, offset, score)


def trusted_suspects(result: "CalibrationResult") -> list[int]:
    """Misplaced board points, unless the lens model itself does not fit.

    A lens model that cannot follow the lens also leaves some points off the same
    way in every image. Then the model is the thing to fix first, and no board
    point is blamed.
    """
    if result.points is None:
        return []
    if result.pattern is not None and result.pattern.has_pattern:
        return []
    return result.points.suspects()
