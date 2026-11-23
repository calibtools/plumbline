"""Several boards that are fixed to each other, solved as one object.

With the boards as views of their own (`boards.together`), every board in every image has a
position of its own: twelve unknowns for an image that shows two boards. When the boards are
fixed to each other, an image has one position, six unknowns, and each further board has one
place on the object, the same in all images. Fewer unknowns for the same points, and an object
that is not flat: the camera is pinned down about twice as well in the simulated tests.

One fit finds the camera, the position of the object in every image and the place of every
further board. The steps are Levenberg-Marquardt; the positions of the images are taken out of
each step, so what is solved is small. The same sums give the uncertainty of the camera and of
the places, with everything else fitted along.

The result has the same layout as the one for boards as views of their own (view `b * images + i`
is board b + 1 in image i), so tables, drawings and exports read it the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import lens
from .boards import BoardPlace, board_places, excluded_views, together
from .calibrate import CalibrationError, CalibrationResult, ModelOptions, ViewResult, calibrate
from .detect import Detection
from .targets import Target
from .uncertainty import Uncertainty, _apply, _values, fitted_names

MOST_STEPS = 60
# The boards are taken to move against each other when fixing them raises the error by this factor and by this much.
LOOSE_FACTOR, LOOSE_PX = 1.5, 0.05


@dataclass
class FixedBoards:
    """What the fit says about the object the boards form."""
    places: list[BoardPlace] = field(default_factory=list)   # boards 2, 3, ... seen from board 1
    rms_free: float = float("nan")      # error with every board as a view of its own, for comparison
    rms_fixed: float = float("nan")
    steps: int = 0

    @property
    def loose(self) -> bool:
        """The images fit clearly worse with the boards fixed: they probably moved against each other."""
        return self.rms_fixed > LOOSE_FACTOR * self.rms_free and self.rms_fixed - self.rms_free > LOOSE_PX

    def to_dict(self) -> dict:
        return {"rms_each_on_its_own_px": self.rms_free, "rms_fixed_px": self.rms_fixed, "loose": self.loose}


def _rot(v) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def _rvec(R) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


class _Fit:
    """blocks[i]: what image i shows, a list of (board, points on that board (n, 3), where they were found (n, 2))."""

    def __init__(self, blocks, K, dist, options: ModelOptions, boards: int, free_places: bool = True):
        self.blocks, self.fisheye = blocks, options.fisheye
        self.K0, self.d0 = np.asarray(K, np.float64), np.asarray(dist, np.float64).reshape(-1).copy()
        self.names = fitted_names(options, len(self.d0))
        self.n_cam = len(self.names)
        self.n_place = 6 * (boards - 1) if free_places else 0
        cam_steps = [1e-3 if n in ("f", "fx", "fy", "cx", "cy") else 1e-6 for n in self.names]
        self.steps = np.array(cam_steps + [1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3] * (self.n_place // 6))
        self.count = 2 * sum(len(o) for block in blocks for _b, o, _p in block)

    def start(self, places) -> np.ndarray:
        cam = _values(self.names, self.K0, self.d0, self.fisheye)
        return np.concatenate([cam] + [np.concatenate([_rvec(R), np.asarray(t, np.float64).ravel()]) for R, t in places][: self.n_place // 6])

    def camera(self, g):
        return _apply(self.names, g[: self.n_cam], self.K0, self.d0, self.fisheye)

    def place(self, g, fixed, b: int):
        if b == 0:
            return np.eye(3), np.zeros(3)
        if self.n_place:
            v = g[self.n_cam + 6 * (b - 1): self.n_cam + 6 * b]
            return _rot(v[:3]), v[3:]
        return fixed[b - 1]

    def image(self, i: int, g, pose, fixed) -> np.ndarray:
        """Projected minus found, for everything image i shows."""
        K, d = self.camera(g)
        out = []
        for b, obj, found in self.blocks[i]:
            R, t = self.place(g, fixed, b)
            out.append((lens.project(obj @ R.T + t, pose[:3], pose[3:], K, d, self.fisheye) - found).ravel())
        return np.concatenate(out)

    def sums(self, g, poses, fixed):
        """The pieces of one step: for the camera and the places, and for each image's own position."""
        ng = len(g)
        A, a = np.zeros((ng, ng)), np.zeros(ng)
        per_image, cost = [], 0.0
        for i in range(len(self.blocks)):
            pose = poses[i]
            r = self.image(i, g, pose, fixed)
            Jg = np.zeros((len(r), ng))
            for k in range(ng):
                d = np.zeros(ng)
                d[k] = self.steps[k]
                Jg[:, k] = (self.image(i, g + d, pose, fixed) - self.image(i, g - d, pose, fixed)) / (2 * self.steps[k])
            Jp = np.zeros((len(r), 6))
            pose_steps = np.array([1e-6] * 3 + [1e-4 * max(1.0, float(np.linalg.norm(pose[3:])))] * 3)
            for k in range(6):
                d = np.zeros(6)
                d[k] = pose_steps[k]
                Jp[:, k] = (self.image(i, g, pose + d, fixed) - self.image(i, g, pose - d, fixed)) / (2 * pose_steps[k])
            A += Jg.T @ Jg
            a += Jg.T @ r
            per_image.append((Jg.T @ Jp, Jp.T @ Jp, Jp.T @ r))
            cost += float(r @ r)
        return A, a, per_image, cost

    def cost(self, g, poses, fixed) -> float:
        return float(sum(np.sum(self.image(i, g, poses[i], fixed) ** 2) for i in range(len(self.blocks))))

    def solve(self, g, poses, fixed=()):
        """Returns g, poses, cost, steps taken, and the matrix whose inverse gives the uncertainties."""
        poses = [np.asarray(p, np.float64).copy() for p in poses]
        A, a, per_image, cost = self.sums(g, poses, fixed)
        damp, steps = 1e-3, 0
        for steps in range(1, MOST_STEPS + 1):
            ng = len(g)
            H, h = A + damp * np.diag(np.diag(A)), a.copy()
            solved = []
            for B, C, c in per_image:
                Cd = C + damp * np.diag(np.diag(C))
                Ci = np.linalg.inv(Cd)
                H -= B @ Ci @ B.T
                h -= B @ Ci @ c
                solved.append((B, Ci, c))
            dg = -np.linalg.solve(H, h) if ng else np.zeros(0)
            g2 = g + dg
            poses2 = [p - Ci @ (c + B.T @ dg) for p, (B, Ci, c) in zip(poses, solved)]
            cost2 = self.cost(g2, poses2, fixed)
            if cost2 < cost:
                gain = (cost - cost2) / max(cost, 1e-30)
                g, poses, damp = g2, poses2, max(damp / 5, 1e-9)
                A, a, per_image, cost = self.sums(g, poses, fixed)
                if gain < 1e-10:
                    break
            else:
                damp *= 5
                if damp > 1e8:
                    break
        reduced = A.copy()
        for B, C, _c in per_image:
            reduced -= B @ np.linalg.solve(C, B.T)
        return g, poses, cost, steps, reduced


def calibrate_fixed(
    per_board: list[list[Detection]],
    target: Target,
    options: ModelOptions | None = None,
    excluded: set[int] | None = None,
) -> CalibrationResult:
    """Calibrate with the boards taken as fixed to each other. `excluded`: images left out, counted from 0."""
    options = options or ModelOptions()
    count, n = len(per_board), len(per_board[0])
    if options.release_board:
        raise CalibrationError("Estimating the shape of the board works with one board per image.")
    left_out = sorted(set(excluded or ()))
    free = calibrate(together(per_board), target, options, excluded_views(set(left_out), n, count))
    if count < 2:
        return free
    by = {v.index: v for v in free.views}
    places = board_places(free, n, count)
    for p in places:
        if p.images == 0:
            raise CalibrationError(
                f"No image shows board {p.board + 1} together with board 1, so its place on the object cannot be found. "
                "Add an image that shows both, or switch off \"fixed to each other\".")
    pts = target.object_points().astype(np.float64)
    images = [i for i in range(n) if any(b * n + i in by for b in range(count))]
    blocks, poses = [], []
    for i in images:
        block = [(b, pts[np.asarray(per_board[b][i].ids).reshape(-1)], np.asarray(per_board[b][i].corners, np.float64).reshape(-1, 2))
                 for b in range(count) if b * n + i in by]
        blocks.append(block)
        b = block[0][0]
        v = by[b * n + i]
        R, t = _rot(v.rvec), np.asarray(v.tvec, np.float64).ravel()
        if b:       # board 1 is not in this image: its position follows from the first board that is
            Rb, tb = places[b - 1].rotation, places[b - 1].translation
            R = R @ Rb.T
            t = t - R @ tb
        poses.append(np.concatenate([_rvec(R), t]))
    fit = _Fit(blocks, free.camera_matrix, free.distortion, options, count)
    try:
        g, poses, cost, steps, reduced = fit.solve(fit.start([(p.rotation, p.translation) for p in places]), poses)
        dof = fit.count - len(g) - 6 * len(images)
        if dof <= 0:
            raise CalibrationError("Too few points for the boards as one object.")
        cov = np.linalg.inv(reduced) * (cost / dof)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError("The boards could not be solved as one object from these images.") from exc
    K, dist = fit.camera(g)

    views, where = [], {}
    for k, i in enumerate(images):
        res = fit.image(k, g, poses[k], ()).reshape(-1, 2)
        Ri, ti = _rot(poses[k][:3]), poses[k][3:]
        at = 0
        for b, obj, _found in blocks[k]:
            part = res[at: at + len(obj)]
            at += len(obj)
            Rb, tb = fit.place(g, (), b)
            sq = (part**2).sum(axis=1)
            where[b * n + i] = ViewResult(b * n + i, by[b * n + i].path, len(obj), float(np.sqrt(sq.mean())), float(np.sqrt(sq.max())),
                                          _rvec(Ri @ Rb), Ri @ tb + ti, part)
    views = [where[v.index] for v in free.views]

    var = np.diag(cov)
    uncertainty = projection = None
    if np.all(np.isfinite(var)) and np.all(var > 0):
        std = np.sqrt(var)
        c = fit.n_cam
        uncertainty = Uncertainty(fit.names, std[:c], cov[:c, :c] / np.outer(std[:c], std[:c]), float(np.sqrt(cost / dof)), int(dof),
                                  float(K[1, 1] / K[0, 0]))
        from .uncertainty import projection_uncertainty

        try:
            projection = projection_uncertainty(K, dist, options, uncertainty, free.image_width, free.image_height)
        except (cv2.error, np.linalg.LinAlgError, ValueError):
            projection = None
        for p in places:
            v = g[c + 6 * (p.board - 1): c + 6 * p.board]
            s = std[c + 6 * (p.board - 1): c + 6 * p.board]
            p.rotation, p.translation = _rot(v[:3]), v[3:].copy()
            p.angle_std_deg = float(np.rad2deg(np.linalg.norm(s[:3])))
            p.place_std_mm = float(np.linalg.norm(s[3:]))
    else:
        for p in places:
            v = g[fit.n_cam + 6 * (p.board - 1): fit.n_cam + 6 * p.board]
            p.rotation, p.translation = _rot(v[:3]), v[3:].copy()

    rms_point = float(np.sqrt(cost / (fit.count / 2)))
    result = CalibrationResult(
        image_width=free.image_width, image_height=free.image_height, camera_matrix=np.asarray(K, np.float64), distortion=dist,
        rms_point=rms_point, views=views, options=options, target=target, excluded=list(free.excluded), skipped=list(free.skipped),
        uncertainty=uncertainty, projection=projection)
    result.fixed = FixedBoards(places, free.rms_point, rms_point, steps)
    flat = together(per_board)
    from .points import point_statistics
    from .residuals import residual_pattern

    result.points = point_statistics(result, flat)
    result.pattern = residual_pattern(result, flat)

    # The trial without the far-off points, with the object as it was just found: one image, one view.
    from . import farpoints

    whole_obj, whole_img = [], []
    for block in blocks:
        whole_obj.append(np.concatenate([obj @ fit.place(g, (), b)[0].T + fit.place(g, (), b)[1] for b, obj, _f in block]))
        whole_img.append(np.concatenate([f for _b, _o, f in block]))

    def again(obj_pts, img_pts, guess):
        trial_blocks = [[(0, np.asarray(o, np.float64).reshape(-1, 3), np.asarray(p, np.float64).reshape(-1, 2))]
                        for o, p in zip(obj_pts, img_pts)]
        trial = _Fit(trial_blocks, guess[0], guess[1], options, 1, free_places=False)
        start = [np.concatenate([np.asarray(x, np.float64).ravel() for x in lens.solve_pose(o, p, guess[0], guess[1], options.fisheye)])
                 for (_b, o, p), in trial_blocks]
        tg, tposes, tcost, _steps, _reduced = trial.solve(trial.start(()), start)
        Kt, dt = trial.camera(tg)
        return (float(np.sqrt(tcost / (trial.count / 2))), Kt, dt, [p[:3] for p in tposes], [p[3:] for p in tposes])

    try:
        result.far = farpoints.look(whole_obj, whole_img, images, (free.image_width, free.image_height), options, K, dist,
                                    [p[:3] for p in poses], [p[3:] for p in poses], uncertainty, again, rms_point)
    except (CalibrationError, np.linalg.LinAlgError, cv2.error, ValueError):
        result.far = None
    return result
