"""The shape of the board, estimated together with the camera.

A calibration takes the board to be flat and exactly the size entered. A
board that sags by a millimetre breaks that quietly: the error barely rises,
and the focal length comes out wrong by percent.

Here every point of the board is allowed to sit where the images say it
sits. The camera is solved with the board as currently known; then each
point is moved to the place that best explains all the images it is in; and
so on, until nothing moves. Where the board lies as a whole and how large it
is cannot be told from images (a larger board further away looks the same),
so the estimated board is kept at the place and overall size of the nominal
one. What remains is its shape: out of flat, or stretched one way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lens

MIN_VIEWS_PER_POINT = 4     # a point in fewer images keeps its nominal place
CLEARLY_BETTER = 0.9        # error must fall below this share of what extra freedom alone would give
MIN_SHAPE_MM = 0.05         # and the shape found must be at least this far from nominal somewhere


@dataclass
class BoardShape:
    points: np.ndarray        # (M, 3) every board point; points not estimated keep their nominal place
    deviation: np.ndarray     # (M, 3) estimated minus nominal, mm
    estimated: np.ndarray     # (M,) which points were estimated
    rms_nominal: float        # error with the board as entered, px per point
    rms_released: float       # error with the board estimated
    expected_ratio: float     # the ratio of the two that the added freedom alone would produce
    rounds: int
    settled: bool             # the estimate stopped moving (False: stopped at the round limit)

    @property
    def out_of_flat_mm(self) -> float:
        """Height range of the estimated points: highest minus lowest."""
        z = self.deviation[self.estimated, 2]
        return float(z.max() - z.min()) if z.size else 0.0

    @property
    def in_plane_mm(self) -> float:
        """Largest shift of a point within the plane of the board."""
        d = self.deviation[self.estimated, :2]
        return float(np.linalg.norm(d, axis=1).max()) if d.size else 0.0

    @property
    def clearly_better(self) -> bool:
        """The images fit a shaped board better than extra freedom alone explains."""
        fits = self.rms_released < CLEARLY_BETTER * self.expected_ratio * self.rms_nominal
        return fits and max(self.out_of_flat_mm, self.in_plane_mm) >= MIN_SHAPE_MM

    def to_dict(self) -> dict:
        return {
            "points_mm": [[float(v) for v in p] for p in self.points],
            "estimated": [bool(v) for v in self.estimated],
            "out_of_flat_mm": self.out_of_flat_mm,
            "in_plane_mm": self.in_plane_mm,
            "rms_with_nominal_board": self.rms_nominal,
            "rms_with_estimated_board": self.rms_released,
            "settled": self.settled,
        }


def keep_place_and_size(points: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    """Move, turn and scale `points` as a whole so that they lie as close to `nominal` as possible."""
    a, b = points - points.mean(axis=0), nominal - nominal.mean(axis=0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    scale = float(np.trace(np.diag(S) @ D) / np.sum(a * a))
    return scale * a @ (U @ D @ Vt) + nominal.mean(axis=0)


def move_points(board, ids_per_view, seen_per_view, poses, K, dist, fisheye):
    """One step: each board point goes where the images it is in put it. Returns the moved board and who moved."""
    count = np.zeros(len(board), int)
    A = np.zeros((len(board), 3, 3))
    g = np.zeros((len(board), 3))
    step = 1e-3
    for ids, seen, (rvec, tvec) in zip(ids_per_view, seen_per_view, poses):
        pts = board[ids]
        J = np.zeros((len(ids), 2, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = step
            J[:, :, k] = (lens.project(pts + e, rvec, tvec, K, dist, fisheye)
                          - lens.project(pts - e, rvec, tvec, K, dist, fisheye)) / (2 * step)
        left = seen - lens.project(pts, rvec, tvec, K, dist, fisheye)
        Jt = np.transpose(J, (0, 2, 1))
        np.add.at(A, ids, Jt @ J)
        np.add.at(g, ids, (Jt @ left[:, :, None])[:, :, 0])
        np.add.at(count, ids, 1)
    moving = count >= MIN_VIEWS_PER_POINT
    out = board.copy()
    if moving.any():
        damp = 1e-9 * np.eye(3)
        out[moving] += np.linalg.solve(A[moving] + damp, g[moving][:, :, None])[:, :, 0]
    return out, moving


def estimate_board(nominal, ids_per_view, seen_per_view, solve, first, fisheye, max_rounds, camera_parameters=10):
    """Alternate between solving the camera and moving the board points.

    nominal: (M, 3) board points as entered. ids_per_view / seen_per_view: for each
    image, which points it shows and where (N, 2). solve(points_per_view, K, dist)
    runs the camera solver from a starting guess and returns rms, K, dist, rvecs, tvecs.
    first: that same tuple for the nominal board.
    Returns (BoardShape, the last solver result).
    """
    nominal = np.asarray(nominal, np.float64)
    board = nominal.copy()
    rms0, K, dist, rvecs, tvecs = first
    last = first
    estimated = np.zeros(len(nominal), bool)
    settled = False
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        moved, estimated = move_points(board, ids_per_view, seen_per_view, list(zip(rvecs, tvecs)), K, dist, fisheye)
        if estimated.sum() < 4:
            break
        moved[estimated] = keep_place_and_size(moved[estimated], nominal[estimated])
        change = float(np.abs(moved - board).max())
        board = moved
        last = solve([board[ids] for ids in ids_per_view], K, dist)
        _rms, K, dist, rvecs, tvecs = last
        if change < 2e-4:  # mm; a looser limit stops while a stretch is still creeping in
            settled = True
            break
    measurements = 2 * sum(len(ids) for ids in ids_per_view)
    before = measurements - camera_parameters - 6 * len(ids_per_view)
    after = before - 3 * int(estimated.sum()) + 7
    ratio = float(np.sqrt(max(after, 1) / max(before, 1)))
    shape = BoardShape(board, board - nominal, estimated, float(rms0), float(last[0]), ratio, rounds, settled)
    return shape, last
