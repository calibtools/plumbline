"""Check a two-camera result with pairs of images that took no part in it.

Every point of the board that both cameras see is placed in space from the two
images alone. The board is then a ruler: the distance between two of its points
is known, and what the cameras measure can be compared with it. The points of a
flat board should also lie in one plane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lens
from .detect import Detection
from .rig import RigResult
from .targets import Target

MIN_POINTS = 6


@dataclass
class PairCheck:
    index: int
    points: int
    length_error_rms_mm: float     # measured minus true, over the long spans of this pair
    length_error_max_mm: float
    flatness_rms_mm: float         # distance of the points from the plane that fits them best
    distance_mm: float             # how far the board was from camera 1
    ray_gap_rms_mm: float          # how closely the two cameras' viewing rays pass each other


@dataclass
class CheckResult:
    pairs: list[PairCheck]
    skipped: list[int]             # pairs without enough points seen by both cameras
    spans: int                     # how many lengths were compared
    longest_mm: float              # the longest of them
    scale: float                   # measured length / true length, median over all spans
    length_error_rms_mm: float
    length_error_max_mm: float
    flatness_rms_mm: float

    def to_dict(self) -> dict:
        return {
            "pairs": [vars(p) for p in self.pairs], "pairs_skipped": list(self.skipped),
            "spans_compared": self.spans, "longest_span_mm": self.longest_mm,
            "measured_over_true": self.scale, "length_error_rms_mm": self.length_error_rms_mm,
            "length_error_max_mm": self.length_error_max_mm, "flatness_rms_mm": self.flatness_rms_mm,
        }


def locate(rig: RigResult, px1: np.ndarray, px2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points in camera 1 coordinates (mm) from where the two cameras see them, and the gap between the rays."""
    a, b = rig.first, rig.second
    d1 = lens.pixel_directions(px1, a.camera_matrix, a.distortion, a.fisheye)
    d2 = lens.pixel_directions(px2, b.camera_matrix, b.distortion, b.fisheye) @ rig.rotation   # into camera 1's frame
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    centre = rig.position                       # camera 2, seen from camera 1
    # Closest points of the two rays s*d1 and centre + t*d2.
    c = np.einsum("ij,ij->i", d1, d2)
    e1, e2 = d1 @ centre, d2 @ centre
    den = np.maximum(1 - c * c, 1e-12)
    s, t = (e1 - c * e2) / den, (c * e1 - e2) / den
    p1, p2 = s[:, None] * d1, centre + t[:, None] * d2
    return (p1 + p2) / 2, np.linalg.norm(p1 - p2, axis=1)


def _measure(board: np.ndarray, least: float, located) -> CheckResult:
    """Lengths and flatness from points placed in space. `located` gives, per pair or set, None or (ids, points, gaps)."""
    pairs, skipped, all_err, all_ratio, all_flat, longest = [], [], [], [], [], 0.0
    for i, found in enumerate(located):
        if found is None:
            skipped.append(i)
            continue
        ids, pts, gap = found
        ok = np.isfinite(pts).all(axis=1)
        pts, gap, ids = pts[ok], gap[ok], ids[ok]
        true = np.linalg.norm(board[ids][:, None] - board[ids][None], axis=2)
        seen = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        pick = np.triu(true >= least, 1)
        if len(ids) < MIN_POINTS or not pick.any():
            skipped.append(i)
            continue
        err = (seen - true)[pick]
        centred = pts - pts.mean(axis=0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        flat = centred @ normal
        pairs.append(PairCheck(i, int(len(ids)), float(np.sqrt(np.mean(err**2))), float(np.abs(err).max()),
                               float(np.sqrt(np.mean(flat**2))), float(np.linalg.norm(pts.mean(axis=0))),
                               float(np.sqrt(np.mean(gap**2)))))
        all_err.append(err)
        all_ratio.append((seen / np.maximum(true, 1e-9))[pick])
        all_flat.append(flat)
        longest = max(longest, float(true[pick].max()))
    if not pairs:
        raise ValueError("None of these images show enough of the target in two cameras to be checked.")
    err, ratio, flat = np.concatenate(all_err), np.concatenate(all_ratio), np.concatenate(all_flat)
    return CheckResult(pairs, skipped, int(len(err)), longest, float(np.median(ratio)),
                       float(np.sqrt(np.mean(err**2))), float(np.abs(err).max()), float(np.sqrt(np.mean(flat**2))))


def check_pairs(rig: RigResult, first: list[Detection], second: list[Detection], target: Target,
                long_share: float = 0.4) -> CheckResult:
    """Compare what the two cameras measure on the board with the board's own dimensions.

    Only spans of at least `long_share` of the board's diagonal are compared: over a short span
    the error of finding the two end points hides everything else.
    """
    board = np.asarray(target.object_points(), np.float64)
    least = long_share * float(np.linalg.norm(board.max(axis=0) - board.min(axis=0)))

    def located():
        for da, db in zip(first, second):
            common = np.intersect1d(da.ids, db.ids) if da.usable and db.usable else np.zeros(0, int)
            if len(common) < MIN_POINTS:
                yield None
                continue
            ia = {int(k): j for j, k in enumerate(np.asarray(da.ids).ravel())}
            ib = {int(k): j for j, k in enumerate(np.asarray(db.ids).ravel())}
            pts, gap = locate(rig, np.asarray(da.corners)[[ia[int(k)] for k in common]],
                              np.asarray(db.corners)[[ib[int(k)] for k in common]])
            yield common, pts, gap

    return _measure(board, least, located())


def check_sets(group, detections: list[list[Detection]], target: Target, long_share: float = 0.4) -> CheckResult:
    """The same check for three or more cameras: every point at least two of them see is placed in space.

    `detections[c][i]` is camera c's image of set i. A point is placed where it is closest to all the
    viewing rays that go to it; its gap is the RMS distance from those rays.
    """
    board = np.asarray(target.object_points(), np.float64)
    least = long_share * float(np.linalg.norm(board.max(axis=0) - board.min(axis=0)))
    places = [(np.eye(3), np.zeros(3))] + [(p.rotation, p.translation) for p in group.placed]

    def located():
        for i in range(len(detections[0])):
            # Per board point: where each ray starts and which way it goes, in camera 1 coordinates.
            rays: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
            for c, cam in enumerate(group.cameras):
                det = detections[c][i] if i < len(detections[c]) else None
                if det is None or not det.usable:
                    continue
                R, T = places[c]
                d = lens.pixel_directions(det.corners, cam.camera_matrix, cam.distortion, cam.fisheye) @ R
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                start = -R.T @ T
                for pid, way in zip(np.asarray(det.ids).ravel(), d):
                    if np.isfinite(way).all():
                        rays.setdefault(int(pid), []).append((start, way))
            ids, pts, gaps = [], [], []
            for pid, seen in sorted(rays.items()):
                if len(seen) < 2:
                    continue
                A = np.zeros((3, 3)); b = np.zeros(3)
                for start, way in seen:
                    P = np.eye(3) - np.outer(way, way)      # what is left of a vector across the ray
                    A += P; b += P @ start
                if np.linalg.cond(A) > 1e9:
                    continue                                 # rays that run alongside each other pin nothing down
                x = np.linalg.solve(A, b)
                off = [np.linalg.norm((np.eye(3) - np.outer(way, way)) @ (x - start)) for start, way in seen]
                ids.append(pid); pts.append(x); gaps.append(float(np.sqrt(np.mean(np.square(off)))))
            yield (np.array(ids, int), np.array(pts), np.array(gaps)) if len(ids) >= MIN_POINTS else None

    return _measure(board, least, located())
