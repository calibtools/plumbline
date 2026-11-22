"""How well the images pin down each camera parameter.

The fit minimises the squared reprojection error. Near the solution the
error grows like a quadratic bowl; a narrow bowl in the direction of a
parameter means the images determine it well. From the shape of that bowl
and the size of the residuals follows a standard deviation for every
parameter, and how strongly two parameters can trade against each other.

What these figures cover: noise in the detected points that is independent
from point to point. What they do not cover: a board that is not flat or not
the size entered, a lens model that does not fit the lens, motion blur that
shifts all points of an image together. They are a lower bound on the real
uncertainty, not the whole of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lens
from .calibrate import ModelOptions

# Position of each optional term in OpenCV's distortion vector.
_STANDARD_SLOTS = {"k1": 0, "k2": 1, "p1": 2, "p2": 3, "k3": 4, "k4": 5, "k5": 6, "k6": 7,
                   "s1": 8, "s2": 9, "s3": 10, "s4": 11}


@dataclass
class Uncertainty:
    names: list[str]            # fitted camera parameters, e.g. fx fy cx cy k1 k2 p1 p2
    std: np.ndarray             # one standard deviation for each
    correlation: np.ndarray     # correlation between them, -1 to 1
    noise_px: float             # point noise per coordinate estimated from the residuals
    degrees_of_freedom: int     # measurements minus fitted quantities
    aspect: float = 1.0         # fy / fx, used when the two were fitted as one

    def std_of(self, name: str) -> float | None:
        """Standard deviation of a parameter, or None if the fit did not vary it."""
        if name in self.names:
            return float(self.std[self.names.index(name)])
        if name in ("fx", "fy") and "f" in self.names:  # one focal length, fy follows fx
            return float(self.std[self.names.index("f")]) * (self.aspect if name == "fy" else 1.0)
        return None

    def strongest_pairs(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Pairs of parameters that can largely stand in for each other, strongest first."""
        out = []
        for i in range(len(self.names)):
            for j in range(i + 1, len(self.names)):
                c = float(self.correlation[i, j])
                if abs(c) >= threshold:
                    out.append((self.names[i], self.names[j], c))
        return sorted(out, key=lambda item: -abs(item[2]))

    def to_dict(self) -> dict:
        return {
            "parameters": self.names,
            "standard_deviation": [float(v) for v in self.std],
            "correlation": [[float(v) for v in row] for row in self.correlation],
            "noise_px_per_coordinate": self.noise_px,
            "degrees_of_freedom": self.degrees_of_freedom,
        }


def fitted_names(options: ModelOptions, distortion_count: int) -> list[str]:
    """The camera parameters the fit was free to change, in display order."""
    names = ["f"] if options.fix_aspect_ratio and not options.fisheye else ["fx", "fy"]
    names += ["cx", "cy"]
    if options.fisheye:
        return names + ["k1", "k2", "k3", "k4"]
    names += ["k1", "k2"]
    if options.tangential:
        names += ["p1", "p2"]
    if options.k3:
        names += ["k3"]
    if options.rational:
        names += ["k4", "k5", "k6"]
    if options.thin_prism:
        names += ["s1", "s2", "s3", "s4"]
    return [n for n in names if n not in _STANDARD_SLOTS or _STANDARD_SLOTS[n] < distortion_count]


def _apply(names, values, K0, dist0, fisheye):
    """Camera matrix and distortion with the named parameters set to the given values."""
    K, dist = K0.copy(), dist0.copy()
    for name, v in zip(names, values):
        if name == "f":
            K[1, 1] = v * K0[1, 1] / K0[0, 0]
            K[0, 0] = v
        elif name == "fx":
            K[0, 0] = v
        elif name == "fy":
            K[1, 1] = v
        elif name == "cx":
            K[0, 2] = v
        elif name == "cy":
            K[1, 2] = v
        elif fisheye:
            dist[int(name[1]) - 1] = v
        else:
            dist[_STANDARD_SLOTS[name]] = v
    return K, dist


def _values(names, K, dist, fisheye):
    out = []
    for name in names:
        if name in ("f", "fx"):
            out.append(K[0, 0])
        elif name == "fy":
            out.append(K[1, 1])
        elif name == "cx":
            out.append(K[0, 2])
        elif name == "cy":
            out.append(K[1, 2])
        elif fisheye:
            out.append(dist[int(name[1]) - 1])
        else:
            out.append(dist[_STANDARD_SLOTS[name]])
    return np.asarray(out, np.float64)


def parameter_uncertainty(K, dist, options: ModelOptions, views) -> Uncertainty | None:
    """Standard deviations and correlations of the fitted camera parameters.

    views: for each image used, (object points (N, 3), rvec, tvec, residuals (N, 2)).
    The board pose of every image is fitted too; its uncertainty is folded
    into the camera parameters, it is not ignored.
    """
    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).reshape(-1)
    fisheye = options.fisheye
    names = fitted_names(options, len(dist))
    theta = _values(names, K, dist, fisheye)
    n_cam = len(names)
    # Steps for the numerical slopes: pixels for focal length and centre, small for lens terms.
    steps = np.array([1e-3 if n in ("f", "fx", "fy", "cx", "cy") else 1e-6 for n in names])

    def proj(values, obj, rvec, tvec):
        Kv, dv = _apply(names, values, K, dist, fisheye)
        return lens.project(obj, rvec, tvec, Kv, dv, fisheye).reshape(-1)

    A = np.zeros((n_cam, n_cam))
    reduced = np.zeros((n_cam, n_cam))
    sse, count = 0.0, 0
    for obj, rvec, tvec, residuals in views:
        obj = np.asarray(obj, np.float64)
        rvec = np.asarray(rvec, np.float64).reshape(3)
        tvec = np.asarray(tvec, np.float64).reshape(3)
        m = 2 * len(obj)
        Jc = np.zeros((m, n_cam))
        for k in range(n_cam):
            d = np.zeros(n_cam)
            d[k] = steps[k]
            Jc[:, k] = (proj(theta + d, obj, rvec, tvec) - proj(theta - d, obj, rvec, tvec)) / (2 * steps[k])
        Jp = np.zeros((m, 6))
        pose = np.concatenate([rvec, tvec])
        pose_steps = np.array([1e-6] * 3 + [1e-4 * max(1.0, float(np.linalg.norm(tvec)))] * 3)
        for k in range(6):
            d = np.zeros(6)
            d[k] = pose_steps[k]
            hi, lo = pose + d, pose - d
            Jp[:, k] = (proj(theta, obj, hi[:3], hi[3:]) - proj(theta, obj, lo[:3], lo[3:])) / (2 * pose_steps[k])
        A += Jc.T @ Jc
        B = Jc.T @ Jp
        C = Jp.T @ Jp
        try:
            reduced += B @ np.linalg.solve(C, B.T)
        except np.linalg.LinAlgError:
            return None
        sse += float(np.sum(np.asarray(residuals, np.float64) ** 2))
        count += m
    dof = count - n_cam - 6 * len(views)
    if dof <= 0:
        return None
    try:
        cov = np.linalg.inv(A - reduced) * (sse / dof)
    except np.linalg.LinAlgError:
        return None
    var = np.diag(cov)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return None
    std = np.sqrt(var)
    return Uncertainty(names, std, cov / np.outer(std, std), float(np.sqrt(sse / dof)), int(dof),
                       float(K[1, 1] / K[0, 0]))
