"""Perspective bias of circle centres.

A circle seen at an angle appears as an ellipse, and the centre of that
ellipse is not the image of the circle's centre: the far half of the circle
is foreshortened more than the near half, so the ellipse centre sits a
little towards the near side. A blob detector returns the ellipse centre.
With the camera and the pose of the board known approximately, the shift
can be computed and taken off.
"""
from __future__ import annotations

import numpy as np

from . import lens

_RIM = 48  # points around each circle


def centre_bias(centres, radius_mm: float, rvec, tvec, K, dist, fisheye: bool = False) -> np.ndarray:
    """For circles at `centres` (N, 3) on the board: image centroid of the circle minus image of its centre.

    The rim of each circle is projected and the centroid of the resulting
    outline is taken as an area centroid, which is what a blob detector measures.
    """
    centres = np.asarray(centres, np.float64).reshape(-1, 3)
    angles = np.linspace(0, 2 * np.pi, _RIM, endpoint=False)
    ring = radius_mm * np.stack([np.cos(angles), np.sin(angles), np.zeros(_RIM)], axis=1)
    rim = (centres[:, None, :] + ring[None, :, :]).reshape(-1, 3)
    outline = lens.project(rim, rvec, tvec, K, dist, fisheye).reshape(len(centres), _RIM, 2)
    x, y = outline[:, :, 0], outline[:, :, 1]
    xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yn - xn * y
    area = cross.sum(axis=1) / 2
    cx = ((x + xn) * cross).sum(axis=1) / (6 * area)
    cy = ((y + yn) * cross).sum(axis=1) / (6 * area)
    middle = lens.project(centres, rvec, tvec, K, dist, fisheye)
    return np.stack([cx, cy], axis=1) - middle
