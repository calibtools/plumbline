"""A look at whether a few far-off points steer the result.

After a calibration the points that lie furthest from the fit are set aside and the camera is
solved once more, as a trial. The result that is reported does not change. When the trial moves
a camera parameter by more than its stated uncertainty, a handful of points is deciding the
answer — a corner found in the wrong place, an image taken while something moved — and the
user is told, with the images those points are in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lens

# A point is far off beyond this many standard deviations of the usual scatter. The scatter of detected
# corners has longer tails than a bell curve: on clean rendered sets a limit of 3.5 sets aside a tenth of
# all points and the trial then moves the camera by two to three of its stated uncertainties; at 5 it still
# does by more than one. At 8 a clean set loses none or one or two points and moves by a third at most,
# while a corner misplaced by 2 px, or an image taken while the board moved, still shows.
FAR = 8.0
ROUNDS = 6
MIN_LEFT = 6     # an image with fewer points left takes no part in the trial


@dataclass
class FarPoints:
    dropped: int                                   # points set aside in the trial
    by_view: dict[int, int] = field(default_factory=dict)   # image index -> points set aside in it
    name: str = "fx"                               # the camera parameter that moves most, against its own uncertainty
    shift: float = 0.0                             # trial value minus reported value, pixels
    std: float = float("nan")                      # the reported uncertainty of that parameter
    rms_point: float = float("nan")                # error of the trial

    @property
    def matters(self) -> bool:
        return self.std == self.std and self.std > 0 and abs(self.shift) > self.std


def look(obj_pts, img_pts, indices, size, options, K, dist, rvecs, tvecs, uncertainty, solve) -> FarPoints | None:
    """Trial solve without the far-off points. `solve(obj, img, guess)` runs the solver once."""
    if uncertainty is None:
        return None
    obj = [np.asarray(o, np.float64).reshape(-1, 3) for o in obj_pts]
    img = [np.asarray(p, np.float64).reshape(-1, 2) for p in img_pts]
    keep = [np.ones(len(o), bool) for o in obj]
    K1, d1, r1, t1 = K, dist, list(rvecs), list(tvecs)
    live = list(range(len(obj)))      # views still in the trial
    rms = float("nan")
    for _ in range(ROUNDS):
        norms = [np.linalg.norm(lens.project(obj[v][keep[v]], r1[j], t1[j], K1, d1, options.fisheye) - img[v][keep[v]], axis=1)
                 for j, v in enumerate(live)]
        # The median of the distances gives the usual scatter without being moved by the far-off points themselves.
        sigma = float(np.median(np.concatenate(norms))) / 1.1774
        changed = False
        for j, v in enumerate(live):
            far = norms[j] > FAR * sigma
            if far.any():
                at = np.nonzero(keep[v])[0][far]
                keep[v][at] = False
                changed = True
        if not changed:
            break
        live = [v for v in live if keep[v].sum() >= MIN_LEFT]
        if len(live) < 4:
            return None
        rms, K1, d1, r1, t1 = solve([obj[v][keep[v]].reshape(-1, 1, 3).astype(np.float32) for v in live],
                                    [img[v][keep[v]].reshape(-1, 1, 2).astype(np.float32) for v in live], (K1, d1))
    dropped = int(sum((~k).sum() for k in keep))
    if not dropped:
        return FarPoints(0)
    values = {"fx": (K[0, 0], K1[0, 0]), "fy": (K[1, 1], K1[1, 1]), "cx": (K[0, 2], K1[0, 2]), "cy": (K[1, 2], K1[1, 2])}
    best = None
    for name, (was, now) in values.items():
        if name not in uncertainty.names:
            continue
        std = float(uncertainty.std[uncertainty.names.index(name)])
        ratio = abs(now - was) / std if std > 0 else 0.0
        if best is None or ratio > best[0]:
            best = (ratio, name, float(now - was), std)
    if best is None:
        return None
    by_view = {int(indices[v]): int((~keep[v]).sum()) for v in range(len(obj)) if (~keep[v]).any()}
    return FarPoints(dropped, by_view, best[1], best[2], best[3], float(rms))
