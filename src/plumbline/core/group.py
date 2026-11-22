"""Three or more cameras that look at the same board: where each sits relative to camera 1.

Each camera is calibrated on its own first. Two cameras are linked when enough sets of
images show the board in both; from camera 1 the links are followed until every camera
has a starting place, so a camera that never shares a view with camera 1 is reached
through one that does. One joint fit over all sets then refines every place at once.
As with two cameras, every point any camera saw counts, and the lens models can differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import lens
from .calibrate import CalibrationError, CalibrationResult, ModelOptions, calibrate
from .detect import Detection
from .rig import MIN_PAIRS, ROUNDS, STEP, _middle, _nudged, _rot, _rvec
from .targets import Target


@dataclass
class Placed:
    """One camera other than camera 1."""
    camera: int                # position in the list of cameras, counted from 0
    rotation: np.ndarray       # 3 x 3: a point in camera 1 coordinates, turned into this camera's coordinates
    translation: np.ndarray    # (3,), mm: x = rotation @ x1 + translation
    through: int               # the camera its starting place was worked out from
    sets: int                  # sets in which this camera and at least one other saw the board
    std: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))            # rotation vector (rad), translation (mm)
    position_std: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))   # mm
    apart_std_mm: float = float("nan")
    turned_std_deg: float = float("nan")

    @property
    def position(self) -> np.ndarray:
        """Where this camera sits, in camera 1 coordinates, mm."""
        return -self.rotation.T @ self.translation

    @property
    def apart_mm(self) -> float:
        return float(np.linalg.norm(self.translation))

    @property
    def turned_deg(self) -> float:
        return float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(self.rotation)[0])))

    def to_dict(self) -> dict:
        return {
            "camera": self.camera + 1,
            "rotation": [[float(v) for v in row] for row in self.rotation],
            "translation_mm": [float(v) for v in self.translation],
            "position_mm": [float(v) for v in self.position],
            "apart_mm": self.apart_mm,
            "turned_deg": self.turned_deg,
            "position_std_mm": [float(v) for v in self.position_std],
            "apart_std_mm": self.apart_std_mm,
            "turned_std_deg": self.turned_std_deg,
            "reached_through_camera": self.through + 1,
            "sets": self.sets,
        }


@dataclass
class SetResult:
    index: int                 # position in every camera's list of detections
    cameras: tuple[int, ...]   # the cameras that saw the board in this set
    rms_point: float           # all of them together, pixels
    points: int


@dataclass
class GroupResult:
    cameras: list[CalibrationResult]
    placed: list[Placed]       # cameras 2, 3, ... in order
    rms_point: float
    sets: list[SetResult]
    unused: list[int]          # sets with the board in fewer than two cameras, or excluded

    def to_dict(self) -> dict:
        return {
            "cameras": [p.to_dict() for p in self.placed],
            "rms_point": self.rms_point,
            "sets_used": [s.index for s in self.sets],
            "sets_not_used": list(self.unused),
        }


class _Fit:
    """Residuals of all sets, for six numbers per camera after the first and a board pose per set in camera 1."""

    def __init__(self, board, sets: list[dict[int, Detection]], cams: list[CalibrationResult]):
        self.board, self.sets, self.cams = board, sets, cams

    def one(self, k: int, rigs: np.ndarray, pose: np.ndarray) -> np.ndarray:
        r1, t1 = pose[:3], pose[3:]
        R1 = _rot(r1)
        out = []
        for c, det in sorted(self.sets[k].items()):
            cam = self.cams[c]
            if c == 0:
                r, t = r1, t1
            else:
                at = 6 * (c - 1)
                R = _rot(rigs[at:at + 3])
                r, t = _rvec(R @ R1), R @ t1 + rigs[at + 3:at + 6]
            p = lens.project(self.board[det.ids], r, t, cam.camera_matrix, cam.distortion, cam.fisheye)
            out.append((p - det.corners).ravel())
        return np.concatenate(out)

    def solve(self, rigs: np.ndarray, poses: np.ndarray):
        """Levenberg-Marquardt. Returns the cameras' numbers, poses, residuals per set, and the covariance."""
        n, m = len(self.sets), len(rigs)
        res = [self.one(k, rigs, poses[k]) for k in range(n)]
        cost = sum(float(r @ r) for r in res)
        damp, cov = 1e-3, None
        for _ in range(ROUNDS):
            # Normal equations in blocks: a camera touches the sets it is in, a pose only its own set.
            A = np.zeros((m, m)); g = np.zeros(m)
            blocks = []
            for k in range(n):
                Jr = np.zeros((len(res[k]), m)); Jp = np.empty((len(res[k]), 6))
                for c in self.sets[k]:
                    for j in range(6 * (c - 1), 6 * c) if c else ():
                        d = np.zeros(m); d[j] = STEP * max(1.0, abs(rigs[j]))
                        Jr[:, j] = (self.one(k, rigs + d, poses[k]) - res[k]) / d[j]
                for j in range(6):
                    d = np.zeros(6); d[j] = STEP * max(1.0, abs(poses[k][j]))
                    Jp[:, j] = (self.one(k, rigs, poses[k] + d) - res[k]) / d[j]
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
                new_rigs = rigs + step
                new_poses = np.array([poses[k] - inv[k] @ (blocks[k][2] + blocks[k][0].T @ step) for k in range(n)])
                new_res = [self.one(k, new_rigs, new_poses[k]) for k in range(n)]
                new_cost = sum(float(r @ r) for r in new_res)
                if new_cost < cost:
                    gain = cost - new_cost
                    rigs, poses, res, cost, damp, improved = new_rigs, new_poses, new_res, new_cost, damp / 5, True
                    break
                damp *= 8
            if not improved or gain < 1e-10 * max(cost, 1e-12):
                break
        S = A.copy()
        for B, C, _gp in blocks:
            S -= B @ np.linalg.inv(C) @ B.T
        dof = sum(len(r) for r in res) - m - 6 * n
        if dof > 0:
            cov = np.linalg.inv(S) * cost / dof
        return rigs, poses, res, cov


def _from_the_cameras(fit: _Fit, rigs: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """What the uncertainty of each camera's own parameters adds: see rig._from_the_cameras."""
    extra = np.zeros((len(rigs), len(rigs)))
    for c, cam in enumerate(fit.cams):
        u = cam.uncertainty
        if u is None:
            continue
        shifts = np.zeros((len(rigs), len(u.names)))
        for j, name in enumerate(u.names):
            moved = _nudged(cam, name, float(u.std[j]))
            if moved is None:
                continue
            other = _Fit(fit.board, fit.sets, [moved if i == c else s for i, s in enumerate(fit.cams)])
            shifts[:, j] = other.solve(rigs, poses)[0] - rigs
        extra += shifts @ u.correlation @ shifts.T
    return extra


def calibrate_group(
    detections: list[list[Detection]],
    target: Target,
    options: list[ModelOptions | None] | None = None,
    excluded: set[int] | None = None,
) -> GroupResult:
    """Calibrate several cameras and where each sits relative to the first.

    `detections[c][i]` is camera c's image of set i. A set counts for how the cameras sit
    to each other when at least two cameras have a usable detection in it.
    """
    count = len(detections)
    if count < 2:
        raise CalibrationError("At least two cameras are needed.")
    if len({len(d) for d in detections}) != 1:
        raise CalibrationError("Every camera needs the same number of images, one per set.")
    excluded = set(excluded or ())
    options = list(options) if options is not None else [None] * count
    cams = []
    for c, (d, o) in enumerate(zip(detections, options)):
        try:
            cams.append(calibrate(d, target, o, excluded))
        except CalibrationError as e:
            raise CalibrationError(f"Camera {c + 1}: {e}") from e
    views = [{v.index: v for v in cam.views} for cam in cams]
    total = len(detections[0])

    # How each camera sits to each other camera it shares enough sets with.
    links = {}
    for a in range(count):
        for b in range(a + 1, count):
            estimates = []
            for i in sorted(set(views[a]) & set(views[b])):
                R = _rot(views[b][i].rvec) @ _rot(views[a][i].rvec).T
                estimates.append((R, np.asarray(views[b][i].tvec, np.float64).ravel()
                                  - R @ np.asarray(views[a][i].tvec, np.float64).ravel()))
            if len(estimates) >= MIN_PAIRS:
                links[(a, b)] = (_middle(estimates), len(estimates))
    place = {0: (np.eye(3), np.zeros(3))}
    through = {}
    while len(place) < count:
        best = None
        for (a, b), (rt, shared) in links.items():
            for src, dst, forward in ((a, b, True), (b, a, False)):
                if src in place and dst not in place and (best is None or shared > best[0]):
                    best = (shared, src, dst, rt, forward)
        if best is None:
            left = ", ".join(str(c + 1) for c in range(count) if c not in place)
            raise CalibrationError(
                f"Camera {left} cannot be placed: fewer than {MIN_PAIRS} sets show the target in it and in a camera that is "
                "already placed." if "," not in left else
                f"Cameras {left} cannot be placed: fewer than {MIN_PAIRS} sets show the target in one of them and in a "
                "camera that is already placed.")
        _shared, src, dst, (R, T), forward = best
        if not forward:
            R, T = R.T, -R.T @ T
        Rs, Ts = place[src]
        place[dst] = (R @ Rs, R @ Ts + T)
        through[dst] = src

    board = np.asarray(target.object_points(), np.float64)
    if all(cam.options.release_board for cam in cams):
        board = np.mean([np.asarray(cam.board_points(), np.float64) for cam in cams], axis=0)

    def seen(det: Detection, view, cam: CalibrationResult):
        """The points as that camera's own solve took them: circle centres already corrected for perspective."""
        ids = np.asarray(det.ids).ravel()
        placed = lens.project(np.asarray(cam.board_points(), np.float64)[ids], view.rvec, view.tvec,
                              cam.camera_matrix, cam.distortion, cam.fisheye)
        return Detection(det.path, det.width, det.height, ids, placed - np.asarray(view.residuals, np.float64))

    sets, used, poses0 = [], [], []
    for i in range(total):
        there = [c for c in range(count) if i in views[c]]
        if len(there) < 2:
            continue
        sets.append({c: seen(detections[c][i], views[c][i], cams[c]) for c in there})
        used.append((i, tuple(there)))
        c = there[0]
        Rc, Tc = place[c]
        poses0.append(np.concatenate([_rvec(Rc.T @ _rot(views[c][i].rvec)),
                                      Rc.T @ (np.asarray(views[c][i].tvec, np.float64).ravel() - Tc)]))
    unused = [i for i in range(total) if i not in {u[0] for u in used}]
    fit = _Fit(board, sets, cams)
    rigs0 = np.concatenate([np.concatenate([_rvec(place[c][0]), place[c][1]]) for c in range(1, count)])
    rigs, poses, res, cov = fit.solve(rigs0, np.array(poses0))
    if cov is not None:
        cov = cov + _from_the_cameras(fit, rigs, poses)

    placed = []
    for c in range(1, count):
        at = 6 * (c - 1)
        v = rigs[at:at + 6]
        out = Placed(c, _rot(v[:3]), v[3:].copy(), through[c], sum(1 for _i, there in used if c in there))
        if cov is not None:
            part = cov[at:at + 6, at:at + 6]
            out.std = np.sqrt(np.clip(np.diag(part), 0, None))
            unit = v[3:] / max(np.linalg.norm(v[3:]), 1e-12)
            out.apart_std_mm = float(np.sqrt(max(unit @ part[3:, 3:] @ unit, 0.0)))
            axis = v[:3] / max(np.linalg.norm(v[:3]), 1e-12)
            out.turned_std_deg = float(np.rad2deg(np.sqrt(max(axis @ part[:3, :3] @ axis, 0.0))))

            def where(w):
                return -_rot(w[:3]).T @ w[3:]

            J = np.empty((3, 6))
            for j in range(6):
                d = np.zeros(6)
                d[j] = 1e-6 * max(1.0, abs(v[j]))
                J[:, j] = (where(v + d) - where(v - d)) / (2 * d[j])
            out.position_std = np.sqrt(np.clip(np.diag(J @ part @ J.T), 0, None))
        placed.append(out)
    results = [SetResult(i, there, float(np.sqrt((r @ r) / (len(r) / 2))), len(r) // 2) for (i, there), r in zip(used, res)]
    points = sum(len(r) // 2 for r in res)
    return GroupResult(cams, placed, float(np.sqrt(sum(float(r @ r) for r in res) / points)), results, unused)
