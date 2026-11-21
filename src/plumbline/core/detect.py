"""Target detection in a single image."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .targets import APRILGRID, APRILGRID_BORDER_BITS, ASYMMETRIC_CIRCLES, CHARUCO, Target

# A view needs enough corners to constrain its pose well.
MIN_CORNERS = 8


@dataclass
class Detection:
    """Corners found in one image.

    ids index into Target.object_points(). corners are (n, 2) pixel
    positions with the origin at the centre of the top-left pixel.
    """

    path: str
    width: int
    height: int
    ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    note: str = ""

    @property
    def count(self) -> int:
        return int(len(self.ids))

    @property
    def usable(self) -> bool:
        return self.count >= MIN_CORNERS

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "ids": self.ids.astype(int).tolist(),
            "corners": np.round(self.corners.astype(float), 4).tolist(),
            "note": self.note,
        }

    @staticmethod
    def from_dict(data: dict) -> "Detection":
        return Detection(
            path=data["path"],
            width=int(data["width"]),
            height=int(data["height"]),
            ids=np.asarray(data["ids"], np.int32).reshape(-1),
            corners=np.asarray(data["corners"], np.float32).reshape(-1, 2),
            note=data.get("note", ""),
        )


def read_gray(path: str) -> np.ndarray | None:
    """Read an image as 8-bit grey. Returns None if it cannot be read."""
    data = np.fromfile(path, dtype=np.uint8)  # handles non-ASCII paths on Windows
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def detect(gray: np.ndarray, target: Target, path: str = "") -> Detection:
    """Find target corners in a grey image."""
    h, w = gray.shape[:2]
    det = Detection(path=path, width=w, height=h)
    if target.kind == CHARUCO:
        _detect_charuco(gray, target, det)
    elif target.kind == APRILGRID:
        _detect_aprilgrid(gray, target, det)
    elif target.is_circles:
        _detect_circles(gray, target, det)
    else:
        _detect_checkerboard(gray, target, det)
    if det.count == 0:
        det.note = det.note or "No target found"
    elif not det.usable:
        det.note = f"Only {det.count} {target.point_word} found, at least {MIN_CORNERS} are needed"
    return det


def detect_file(path: str, target: Target) -> Detection:
    gray = read_gray(path)
    if gray is None:
        return Detection(path=path, width=0, height=0, note="Could not read this file")
    return detect(gray, target, path)


def _detect_charuco(gray: np.ndarray, target: Target, det: Detection) -> None:
    detector = cv2.aruco.CharucoDetector(target.charuco_board())
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if ids is None or len(ids) == 0:
        if marker_ids is not None and len(marker_ids) > 0:
            seen = np.asarray(marker_ids).reshape(-1)
            lo, hi = int(seen.min()), int(seen.max())
            first = target.first_marker_id
            last = first + target.marker_count - 1
            if hi < first or lo > last:
                det.note = (
                    f"The markers in this image have IDs {lo} to {hi}, but the target is set to "
                    f"IDs {first} to {last}. This looks like another board of a set: "
                    f"try First marker ID {lo - lo % target.marker_count}."
                )
            else:
                det.note = (
                    f"{len(seen)} markers found but no board corners. "
                    "Check the square count, the sizes and the pattern layout."
                )
        return
    det.ids = np.asarray(ids, np.int32).reshape(-1)
    det.corners = np.asarray(corners, np.float32).reshape(-1, 2)


def _detect_checkerboard(gray: np.ndarray, target: Target, det: Detection) -> None:
    size = (target.inner_cols, target.inner_rows)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
    if not found or corners is None:
        return
    corners = np.asarray(corners, np.float32).reshape(-1, 2)
    if len(corners) != target.corner_count:
        return
    det.ids = np.arange(target.corner_count, dtype=np.int32)
    det.corners = corners


def _detect_circles(gray: np.ndarray, target: Target, det: Detection) -> None:
    """Dark circles on a light board. The whole grid has to be in view."""
    h, w = gray.shape[:2]
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 12.0
    params.maxArea = max(5000.0, w * h / 40.0)  # the default upper limit rejects circles in large images
    params.minDistBetweenBlobs = 2.0
    blobs = cv2.SimpleBlobDetector_create(params)
    flag = cv2.CALIB_CB_ASYMMETRIC_GRID if target.kind == ASYMMETRIC_CIRCLES else cv2.CALIB_CB_SYMMETRIC_GRID
    found, centres = cv2.findCirclesGrid(gray, target.opencv_size, flags=flag, blobDetector=blobs)
    if not found or centres is None:
        seen = len(blobs.detect(gray))
        if seen:
            det.note = (f"{seen} dark spots found but no complete grid of {target.corner_count} circles. "
                        "The whole grid must be in the image; check the row and column counts.")
        return
    centres = np.asarray(centres, np.float32).reshape(-1, 2)
    if len(centres) != target.corner_count:
        return
    det.ids = np.arange(target.corner_count, dtype=np.int32)
    det.corners = centres


def _detect_aprilgrid(gray: np.ndarray, target: Target, det: Detection) -> None:
    """Tags of an AprilGrid; every tag found gives its four outer corners. Part of the grid is enough."""
    dictionary = target.aruco_dictionary()
    found_ids, found_corners = None, None
    # Kalibr's border is two cells wide; plain AprilTag prints use one.
    for border in (APRILGRID_BORDER_BITS, 1):
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = border
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _rejected = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            found_ids, found_corners = np.asarray(ids).reshape(-1), corners
            break
    if found_ids is None:
        return
    first, count = target.first_marker_id, target.marker_count
    keep = [(int(i) - first, np.asarray(c, np.float32).reshape(4, 2))
            for i, c in zip(found_ids, found_corners) if first <= int(i) < first + count]
    if not keep:
        lo, hi = int(found_ids.min()), int(found_ids.max())
        det.note = (f"The tags in this image have IDs {lo} to {hi}, but the target is set to "
                    f"IDs {first} to {first + count - 1}. Check First tag ID and the number of tags.")
        return
    keep.sort(key=lambda item: item[0])
    keep = [(t, refine_tag_corners(gray, c)) for t, c in keep]
    det.ids = np.concatenate([4 * t + np.arange(4, dtype=np.int32) for t, _c in keep]).astype(np.int32)
    det.corners = np.concatenate([c for _t, c in keep]).astype(np.float32)


def _edge_points(gray: np.ndarray, a: np.ndarray, b: np.ndarray, centre: np.ndarray,
                 t0: float, t1: float) -> np.ndarray:
    """Sub-pixel points on the edge of a tag between a and b, taken along the stretch t0..t1.

    At each sample a short profile is read across the edge; the edge is where the
    brightness passes halfway between the black tag and the white board. That
    point does not move when the image is blurred evenly, unlike a corner.
    """
    length = float(np.linalg.norm(b - a))
    if length < 12:
        return np.zeros((0, 2), np.float32)
    along = (b - a) / length
    normal = np.array([-along[1], along[0]], np.float32)
    if np.dot(centre - (a + b) / 2, normal) > 0:
        normal = -normal  # point away from the tag
    reach = float(np.clip(length * 0.10, 3.0, 7.0))
    count = int(np.clip(length * (t1 - t0) / 1.5, 4, 24))
    ts = np.linspace(t0, t1, count, dtype=np.float32)
    offs = np.arange(-reach, reach + 0.25, 0.5, dtype=np.float32)
    base = a[None, :] + ts[:, None] * (b - a)[None, :]
    xs = base[:, 0:1] + offs[None, :] * normal[0]
    ys = base[:, 1:2] + offs[None, :] * normal[1]
    prof = cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    ends = max(2, len(offs) // 5)
    dark = prof[:, :ends].mean(axis=1)
    bright = prof[:, -ends:].mean(axis=1)
    contrast = bright - dark
    if contrast.size == 0:
        return np.zeros((0, 2), np.float32)
    good = contrast > max(20.0, 0.5 * float(np.median(contrast)))
    out = []
    for i in np.nonzero(good)[0]:
        half = (dark[i] + bright[i]) / 2
        row = prof[i]
        above = np.nonzero((row[:-1] < half) & (row[1:] >= half))[0]
        if len(above) != 1:
            continue  # not a clean step from black to white
        j = int(above[0])
        frac = (half - row[j]) / max(1e-6, row[j + 1] - row[j])
        d = offs[j] + frac * (offs[j + 1] - offs[j])
        out.append(base[i] + d * normal)
    return np.asarray(out, np.float32).reshape(-1, 2)


ORDER, MARGIN_PX, MARGIN_T = 2, 3.0, 0.08  # curve order; how far from the corner the edge is read: pixels and share of its length


class _Edge:
    """One side of a tag as a gently curved line: offset v = c0 + c1 u + c2 u^2 from the chord."""

    def __init__(self, gray: np.ndarray, a: np.ndarray, b: np.ndarray, centre: np.ndarray):
        self.a = a.astype(np.float64)
        length = float(np.linalg.norm(b - a))
        self.along = (b - a).astype(np.float64) / max(length, 1e-9)
        self.normal = np.array([-self.along[1], self.along[0]])
        margin = min(0.2, max(MARGIN_PX / max(length, 1e-9), MARGIN_T))
        pts = _edge_points(gray, a, b, centre, margin, 1 - margin).astype(np.float64)
        self.ok = len(pts) >= 8
        if self.ok:
            u = (pts - self.a) @ self.along
            v = (pts - self.a) @ self.normal
            # A lens bends a straight edge a little; the higher terms follow that bend.
            self.coef = np.polyfit(u, v, ORDER if len(pts) >= 4 * ORDER + 4 else 1)

    def local(self, point: np.ndarray) -> tuple[float, float]:
        d = point - self.a
        return float(d @ self.along), float(d @ self.normal)

    def at(self, u: float) -> np.ndarray:
        return self.a + u * self.along + float(np.polyval(self.coef, u)) * self.normal

    def miss(self, point: np.ndarray) -> float:
        """How far a point is from this edge, measured across it."""
        u, v = self.local(point)
        return v - float(np.polyval(self.coef, u))


def refine_tag_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Tag corners as the crossing of the two tag edges that meet there.

    A tag's outer corner is an L-shaped corner. Blur rounds it off, and corner
    refinement then lands inside the tag by a fraction of a pixel. The edges
    next to the corner do not have that problem, so each edge is measured along
    its length and each corner is taken where two neighbouring edges cross.
    A corner whose edges cannot be measured keeps the position it came with.
    """
    corners = np.asarray(corners, np.float32).reshape(4, 2)
    centre = corners.mean(axis=0)
    edges = [_Edge(gray, corners[i], corners[(i + 1) % 4], centre) for i in range(4)]
    out = corners.copy()
    for i in range(4):
        leaving, arriving = edges[i], edges[(i - 1) % 4]  # the edge that starts here and the one that ends here
        if not (leaving.ok and arriving.ok):
            continue
        # Walk along the leaving edge to where it meets the arriving one (secant steps from the corner).
        u0, u1 = 0.0, 1.0
        f0, f1 = arriving.miss(leaving.at(u0)), arriving.miss(leaving.at(u1))
        for _ in range(8):
            if abs(f1 - f0) < 1e-12:
                break
            u0, u1, f0 = u1, u1 - f1 * (u1 - u0) / (f1 - f0), f1
            f1 = arriving.miss(leaving.at(u1))
            if abs(f1) < 1e-4:
                break
        point = leaving.at(u1)
        if abs(f1) < 1e-2 and np.linalg.norm(point - corners[i]) <= 3.0:  # a refinement moves a corner a little
            out[i] = point.astype(np.float32)
    return out
