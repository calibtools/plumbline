"""Two cameras that look at the same board: how the second sits relative to the first.

Each camera is calibrated on its own first. Every pair of images taken at the same
moment then gives one estimate of where camera 2 sits; a middle one of these is the
starting point, and a joint fit over all pairs refines it. The fit uses every point
either camera saw, so a board that is partly out of view in one camera still counts,
and it works the same way for both lens models.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from . import lens
from .calibrate import CalibrationError, CalibrationResult, ModelOptions, calibrate
from .detect import Detection
from .targets import Target

MIN_PAIRS = 3
ROUNDS = 30
STEP = 1e-6


@dataclass
class PairResult:
    index: int            # position in the two lists of detections
    rms_point: float      # both images of the pair together, pixels
    points: int           # points seen by camera 1 plus points seen by camera 2


@dataclass
class RigResult:
    first: CalibrationResult
    second: CalibrationResult
    rotation: np.ndarray       # 3 x 3: a point in camera 1 coordinates, turned into camera 2 coordinates
    translation: np.ndarray    # (3,), mm: x2 = rotation @ x1 + translation
    rms_point: float           # all pairs, both cameras, pixels
    pairs: list[PairResult]
    unpaired: list[int]        # pairs where one side has no usable detection, or the user excluded it
    std: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))  # rotation vector (rad), translation (mm)
    apart_std_mm: float = float("nan")
    turned_std_deg: float = float("nan")

    @property
    def position(self) -> np.ndarray:
        """Where camera 2 sits, in camera 1 coordinates, mm."""
        return -self.rotation.T @ self.translation

    @property
    def apart_mm(self) -> float:
        return float(np.linalg.norm(self.translation))

    @property
    def turned_deg(self) -> float:
        return float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(self.rotation)[0])))

    def to_dict(self) -> dict:
        return {
            "rotation": [[float(v) for v in row] for row in self.rotation],
            "translation_mm": [float(v) for v in self.translation],
            "camera_2_position_mm": [float(v) for v in self.position],
            "apart_mm": self.apart_mm,
            "turned_deg": self.turned_deg,
            "apart_std_mm": self.apart_std_mm,
            "turned_std_deg": self.turned_std_deg,
            "rms_point": self.rms_point,
            "pairs_used": [p.index for p in self.pairs],
            "pairs_not_used": list(self.unpaired),
        }


def _rot(rvec) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]


def _rvec(R) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def _middle(estimates: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The estimate closest to all the others: one bad pair cannot pull it away."""
    best, least = 0, np.inf
    for i, (Ri, Ti) in enumerate(estimates):
        total = 0.0
        for Rj, Tj in estimates:
            angle = np.linalg.norm(_rvec(Ri @ Rj.T))
            total += angle + np.linalg.norm(Ti - Tj) / max(np.linalg.norm(Ti), 1.0)
        if total < least:
            best, least = i, total
    return estimates[best]


class _Fit:
    """Residuals of all pairs for a given rig (6 numbers) and board pose per pair in camera 1 (6 each)."""

    def __init__(self, board, pairs, a: CalibrationResult, b: CalibrationResult):
        self.board, self.pairs, self.a, self.b = board, pairs, a, b

    def pair(self, k: int, rig: np.ndarray, pose: np.ndarray) -> np.ndarray:
        da, db = self.pairs[k]
        r1, t1 = pose[:3], pose[3:]
        R, T = _rot(rig[:3]), rig[3:]
        R1 = _rot(r1)
        r2, t2 = _rvec(R @ R1), R @ t1 + T
        pa = lens.project(self.board[da.ids], r1, t1, self.a.camera_matrix, self.a.distortion, self.a.fisheye)
        pb = lens.project(self.board[db.ids], r2, t2, self.b.camera_matrix, self.b.distortion, self.b.fisheye)
        return np.concatenate([(pa - da.corners).ravel(), (pb - db.corners).ravel()])

    def solve(self, rig: np.ndarray, poses: np.ndarray):
        """Levenberg-Marquardt. Returns rig, poses, residuals per pair, and the covariance of the rig."""
        n = len(self.pairs)
        res = [self.pair(k, rig, poses[k]) for k in range(n)]
        cost = sum(float(r @ r) for r in res)
        damp, cov = 1e-3, None
        for _ in range(ROUNDS):
            # Normal equations in blocks: the rig touches every pair, a pose only its own pair.
            A = np.zeros((6, 6)); g = np.zeros(6)
            blocks = []
            for k in range(n):
                Jr = np.empty((len(res[k]), 6)); Jp = np.empty((len(res[k]), 6))
                for c in range(6):
                    d = np.zeros(6); d[c] = STEP * max(1.0, abs(rig[c]))
                    Jr[:, c] = (self.pair(k, rig + d, poses[k]) - res[k]) / d[c]
                    d = np.zeros(6); d[c] = STEP * max(1.0, abs(poses[k][c]))
                    Jp[:, c] = (self.pair(k, rig, poses[k] + d) - res[k]) / d[c]
                A += Jr.T @ Jr; g += Jr.T @ res[k]
                blocks.append((Jr.T @ Jp, Jp.T @ Jp, Jp.T @ res[k]))
            improved = False
            for _try in range(8):
                S = A + damp * np.diag(np.diag(A)); rhs = g.copy()
                inv = []
                for B, C, gp in blocks:
                    Ci = np.linalg.inv(C + damp * np.diag(np.diag(C)))
                    inv.append(Ci); S -= B @ Ci @ B.T; rhs -= B @ Ci @ gp
                step = -np.linalg.solve(S, rhs)
                new_rig = rig + step
                new_poses = np.array([poses[k] - inv[k] @ (blocks[k][2] + blocks[k][0].T @ step) for k in range(n)])
                new_res = [self.pair(k, new_rig, new_poses[k]) for k in range(n)]
                new_cost = sum(float(r @ r) for r in new_res)
                if new_cost < cost:
                    gain = cost - new_cost
                    rig, poses, res, cost, damp, improved = new_rig, new_poses, new_res, new_cost, damp / 5, True
                    break
                damp *= 8
            if not improved or gain < 1e-10 * max(cost, 1e-12):
                break
        # Covariance of the rig with the poses solved out, without damping.
        S = A.copy()
        for B, C, _gp in blocks:
            S -= B @ np.linalg.inv(C) @ B.T
        count = sum(len(r) for r in res)
        dof = count - 6 - 6 * n
        if dof > 0:
            cov = np.linalg.inv(S) * cost / dof
        return rig, poses, res, cov


def _nudged(result: CalibrationResult, name: str, by: float) -> CalibrationResult | None:
    """The same camera with one fitted parameter moved by `by`."""
    K, dist = result.camera_matrix.copy(), np.asarray(result.distortion, np.float64).copy()
    spot = {"fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)}
    if name == "f":
        K[0, 0] += by
        K[1, 1] += by * result.uncertainty.aspect
    elif name in spot:
        K[spot[name]] += by
    elif name in result.distortion_names:
        dist[list(result.distortion_names).index(name)] += by
    else:
        return None
    return replace(result, camera_matrix=K, distortion=dist)


def _from_the_cameras(fit: _Fit, rig: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """What the uncertainty of each camera's own parameters adds to the covariance of the rig.

    Each fitted parameter is moved by one standard deviation and the rig solved again; the
    shifts, combined through the correlations between the parameters, give the added spread.
    """
    extra = np.zeros((6, 6))
    for which in ("a", "b"):
        cam = getattr(fit, which)
        u = cam.uncertainty
        if u is None:
            continue
        shifts = np.zeros((6, len(u.names)))
        for j, name in enumerate(u.names):
            moved = _nudged(cam, name, float(u.std[j]))
            if moved is None:
                continue
            other = _Fit(fit.board, fit.pairs, moved if which == "a" else fit.a, moved if which == "b" else fit.b)
            shifts[:, j] = other.solve(rig, poses)[0] - rig
        extra += shifts @ u.correlation @ shifts.T
    return extra


def calibrate_pair(
    first: list[Detection],
    second: list[Detection],
    target: Target,
    options_first: ModelOptions | None = None,
    options_second: ModelOptions | None = None,
    excluded: set[int] | None = None,
) -> RigResult:
    """Calibrate two cameras and how they sit to each other.

    `first[i]` and `second[i]` are the two images of pair i. An image without a usable
    detection still counts for its own camera if the other one of the pair is good; it
    does not count for how the cameras sit to each other.
    """
    if len(first) != len(second):
        raise CalibrationError("Both cameras need the same number of images, one per pair.")
    excluded = set(excluded or ())
    a = calibrate(first, target, options_first, excluded)
    b = calibrate(second, target, options_second if options_second is not None else options_first, excluded)
    va = {v.index: v for v in a.views}
    vb = {v.index: v for v in b.views}
    both = sorted(set(va) & set(vb))
    unpaired = [i for i in range(len(first)) if i not in both]
    if len(both) < MIN_PAIRS:
        raise CalibrationError(f"Only {len(both)} pairs have the target in both images; at least {MIN_PAIRS} are needed.")

    # The board as the single-camera solves left it: as entered, or with its shape estimated. Each camera
    # estimates the shape on its own; both are held to the same place and size, so their mean is the board.
    board = np.asarray(target.object_points(), np.float64)
    if a.options.release_board and b.options.release_board:
        board = (np.asarray(a.board_points(), np.float64) + np.asarray(b.board_points(), np.float64)) / 2
    estimates = []
    for i in both:
        R1, R2 = _rot(va[i].rvec), _rot(vb[i].rvec)
        R = R2 @ R1.T
        estimates.append((R, np.asarray(vb[i].tvec, np.float64).ravel() - R @ np.asarray(va[i].tvec, np.float64).ravel()))
    R0, T0 = _middle(estimates)

    def seen(det: Detection, view, cam: CalibrationResult):
        """The points as that camera's own solve took them: circle centres already corrected for perspective."""
        ids = np.asarray(det.ids).ravel()
        placed = lens.project(np.asarray(cam.board_points(), np.float64)[ids], view.rvec, view.tvec,
                              cam.camera_matrix, cam.distortion, cam.fisheye)
        return Detection(det.path, det.width, det.height, ids, placed - np.asarray(view.residuals, np.float64))

    fit = _Fit(board, [(seen(first[i], va[i], a), seen(second[i], vb[i], b)) for i in both], a, b)
    rig0 = np.concatenate([_rvec(R0), T0])
    poses0 = np.array([np.concatenate([np.asarray(va[i].rvec, np.float64).ravel(), np.asarray(va[i].tvec, np.float64).ravel()])
                       for i in both])
    rig, _poses, res, cov = fit.solve(rig0, poses0)

    pairs = [PairResult(i, float(np.sqrt((r @ r) / (len(r) / 2))), len(r) // 2) for i, r in zip(both, res)]
    total = sum(len(r) // 2 for r in res)
    out = RigResult(a, b, _rot(rig[:3]), rig[3:].copy(), float(np.sqrt(sum(float(r @ r) for r in res) / total)), pairs, unpaired)
    if cov is not None:
        cov = cov + _from_the_cameras(fit, rig, _poses)
        out.std = np.sqrt(np.clip(np.diag(cov), 0, None))
        T = out.translation
        unit = T / max(np.linalg.norm(T), 1e-12)
        out.apart_std_mm = float(np.sqrt(unit @ cov[3:, 3:] @ unit))
        axis = rig[:3] / max(np.linalg.norm(rig[:3]), 1e-12)
        out.turned_std_deg = float(np.rad2deg(np.sqrt(axis @ cov[:3, :3] @ axis)))
    return out
