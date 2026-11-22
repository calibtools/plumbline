"""What the residuals say beyond their size.

If the lens model fits and the board is as entered, what is left over after
the fit is noise: it points in no particular direction and looks the same
everywhere in the image. A pattern gives a poor fit away even when the RMS
looks fine: in some part of the image the leftovers all lean the same way,
because the lens model cannot follow the lens there.

The shape of the distribution is recorded too (excess kurtosis: how much a
few far-off points dominate), but nothing is concluded from it yet. Rendered
test images already show values between 2 and 19 with nothing wrong, so a
sensible limit has to wait for photographs of real boards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .detect import Detection

if TYPE_CHECKING:
    from .calibrate import CalibrationResult

MIN_PER_CELL = 8          # cells with fewer points are left blank
PATTERN_STRENGTH = 3.0    # cell averages this many times larger (squared) than noise alone would give
PATTERN_PX = 0.1          # and at least this large somewhere, in px


@dataclass
class ResidualPattern:
    xs: np.ndarray          # pixel x of the cell centres, (cols,)
    ys: np.ndarray          # pixel y of the cell centres, (rows,)
    mean: np.ndarray        # (rows, cols, 2) average of observed minus predicted, px; NaN where too few points
    count: np.ndarray       # (rows, cols) points per cell
    strength: float         # 1 = what noise alone gives; larger = the averages are not noise
    largest_px: float       # the largest cell average
    kurtosis: float         # excess kurtosis of all residual coordinates; 0 for a normal distribution

    @property
    def has_pattern(self) -> bool:
        return self.strength > PATTERN_STRENGTH and self.largest_px > PATTERN_PX

    def to_dict(self) -> dict:
        return {
            "x": [float(v) for v in self.xs],
            "y": [float(v) for v in self.ys],
            "mean_px": [[None if not np.isfinite(c[0]) else [float(c[0]), float(c[1])] for c in row]
                        for row in self.mean],
            "strength": self.strength,
            "largest_px": self.largest_px,
            "excess_kurtosis": self.kurtosis,
        }


def residual_pattern(result: "CalibrationResult", detections: list[Detection], cols: int = 12) -> ResidualPattern:
    width, height = result.image_width, result.image_height
    rows = max(2, int(round(cols * height / width)))
    xs = (np.arange(cols) + 0.5) * width / cols
    ys = (np.arange(rows) + 0.5) * height / rows
    where = np.concatenate([np.asarray(detections[v.index].corners, np.float64).reshape(-1, 2) for v in result.views])
    left = -np.concatenate([np.asarray(v.residuals, np.float64) for v in result.views])   # observed minus predicted
    c = np.clip((where[:, 0] * cols / width).astype(int), 0, cols - 1)
    r = np.clip((where[:, 1] * rows / height).astype(int), 0, rows - 1)
    flat = r * cols + c
    count = np.bincount(flat, minlength=rows * cols)
    sums = np.stack([np.bincount(flat, weights=left[:, k], minlength=rows * cols) for k in range(2)], axis=1)
    mean = np.full((rows * cols, 2), np.nan)
    enough = count >= MIN_PER_CELL
    mean[enough] = sums[enough] / count[enough, None]

    noise2 = float(np.mean(left**2))                        # per coordinate
    if enough.any() and noise2 > 0:
        z2 = count[enough] * np.sum(mean[enough] ** 2, axis=1) / noise2    # about chi-square with 2 dof under noise
        strength = float(np.mean(z2) / 2)
        largest = float(np.max(np.linalg.norm(mean[enough], axis=1)))
    else:
        strength, largest = 0.0, 0.0
    v = left.reshape(-1)
    v = v - v.mean()
    m2 = float(np.mean(v**2))
    kurt = float(np.mean(v**4) / m2**2 - 3) if m2 > 0 else 0.0
    return ResidualPattern(xs, ys, mean.reshape(rows, cols, 2), count.reshape(rows, cols), strength, largest, kurt)
