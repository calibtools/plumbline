"""Row alignment for two cameras: after it, a point shows on the same image row in both.

Both images are redrawn as two ideal cameras that look the same way and sit side by
side (or one above the other). Then a point's two images differ only along the row,
by an amount that says how far away it is. This is what stereo matching needs.

The directions the two ideal cameras look in come from OpenCV's stereoRectify. The
lookup tables are built here, by following each pixel's viewing ray back into the
real camera through that camera's own lens model, so a standard lens and a fisheye
can be paired.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import VERSION_NAME
from . import lens
from .rig import RigResult


@dataclass
class Rectification:
    size: tuple[int, int]        # width, height of both aligned images
    R1: np.ndarray               # turns camera 1 coordinates into aligned camera 1 coordinates
    R2: np.ndarray
    P1: np.ndarray               # 3 x 4 projection of aligned camera 1
    P2: np.ndarray
    Q: np.ndarray                # 4 x 4: (column, row, disparity, 1) to a point in aligned camera 1, mm
    vertical: bool               # True when the cameras sit one above the other: points then share a column
    map1: tuple[np.ndarray, np.ndarray] | None = None   # for every aligned pixel, where to read in camera 1's image
    map2: tuple[np.ndarray, np.ndarray] | None = None

    def to_dict(self) -> dict:
        return {
            "note": "Two ideal cameras that look the same way. R1 and R2 turn each camera's coordinates into its "
                    "aligned camera's; P1 and P2 project from there; Q turns (column, row, disparity, 1) into a "
                    "point in aligned camera 1, mm.",
            "image_size": [int(self.size[0]), int(self.size[1])],
            "aligned_along": "columns" if self.vertical else "rows",
            **{k: [[float(v) for v in row] for row in getattr(self, k)] for k in ("R1", "R2", "P1", "P2", "Q")},
        }


def _table(cam, R: np.ndarray, P: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    K = P[:3, :3]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1).reshape(-1, 3)
    in_camera = rays @ R            # R transposed, applied to row vectors: back into the real camera's directions
    ahead = in_camera[:, 2] > 1e-6
    place = np.full((len(rays), 2), -1.0)
    if ahead.any():
        place[ahead] = lens.project(in_camera[ahead], np.zeros(3), np.zeros(3), cam.camera_matrix, cam.distortion,
                                    cam.fisheye)
    place[~np.isfinite(place).all(axis=1)] = -1.0
    return (place[:, 0].reshape(h, w).astype(np.float32), place[:, 1].reshape(h, w).astype(np.float32))


def rectify(rig: RigResult, tables: bool = True) -> Rectification:
    """The aligned pair of cameras. `tables=False` leaves the lookup tables out, which is much quicker."""
    a, b = rig.first, rig.second
    size = (a.image_width, a.image_height)
    none = np.zeros(5)
    # With no distortion handed over, OpenCV works out the two viewing directions and a common pinhole
    # camera from the positions alone; the lenses come in through the lookup tables.
    R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
        np.asarray(a.camera_matrix, np.float64), none, np.asarray(b.camera_matrix, np.float64), none, size,
        np.asarray(rig.rotation, np.float64), np.asarray(rig.translation, np.float64).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    out = Rectification(size, R1, R2, P1, P2, Q, bool(vertical))
    if tables:
        out.map1, out.map2 = _table(a, R1, P1, size), _table(b, R2, P2, size)
    return out


def write_rectify_maps(rig: RigResult, path: str | Path) -> Rectification:
    """Lookup tables for both cameras and the matrices that go with them, as a NumPy .npz file."""
    r = rectify(rig)
    np.savez_compressed(
        str(path),
        map1_x=r.map1[0], map1_y=r.map1[1], map2_x=r.map2[0], map2_y=r.map2[1],
        R1=r.R1, R2=r.R2, P1=r.P1, P2=r.P2, Q=r.Q,
        image_size=np.array(r.size, np.int32),
        aligned_along=np.array("columns" if r.vertical else "rows"),
        generator=np.array(f"Plumbline {VERSION_NAME} by Calib.Tools, https://calib.tools"),
    )
    return r
