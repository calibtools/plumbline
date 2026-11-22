"""Single-camera calibration with OpenCV's standard or fisheye lens model."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import lens
from .detect import Detection
from .targets import Target

MIN_VIEWS = 4
COVERAGE_GRID = (16, 12)  # columns, rows


@dataclass(frozen=True)
class ModelOptions:
    """Which terms of the OpenCV model are estimated."""

    k3: bool = False
    tangential: bool = True
    rational: bool = False  # k4, k5, k6
    thin_prism: bool = False  # s1..s4
    fix_aspect_ratio: bool = False
    fisheye: bool = False  # OpenCV fisheye model, k1..k4; the terms above do not apply

    def flags(self) -> int:
        flags = 0
        if not self.k3:
            flags |= cv2.CALIB_FIX_K3
        if not self.tangential:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        if self.rational:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if self.thin_prism:
            flags |= cv2.CALIB_THIN_PRISM_MODEL
        if self.fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        return flags

    def to_dict(self) -> dict:
        return {
            "k3": self.k3,
            "tangential": self.tangential,
            "rational": self.rational,
            "thin_prism": self.thin_prism,
            "fix_aspect_ratio": self.fix_aspect_ratio,
            "fisheye": self.fisheye,
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelOptions":
        known = ModelOptions.__dataclass_fields__
        return ModelOptions(**{k: bool(v) for k, v in data.items() if k in known})


@dataclass
class ViewResult:
    index: int  # position in the list of detections given to calibrate()
    path: str
    corner_count: int
    rms_point: float  # sqrt(mean(dx^2 + dy^2)), pixels
    max_error: float
    rvec: np.ndarray
    tvec: np.ndarray
    residuals: np.ndarray  # (n, 2) projected minus detected, pixels


@dataclass
class CalibrationResult:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray  # 3 x 3
    # Standard model, OpenCV order: k1 k2 p1 p2 k3 [k4 k5 k6 [s1 s2 s3 s4]]. Fisheye: k1 k2 k3 k4.
    distortion: np.ndarray
    rms_point: float
    views: list[ViewResult]
    options: ModelOptions
    target: Target
    excluded: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    opencv_version: str = cv2.__version__
    uncertainty: "object | None" = None  # core.uncertainty.Uncertainty; None when it cannot be worked out
    points: "object | None" = None        # core.points.PointStats: every board point across its images
    circle_shift_px: float | None = None  # mean perspective correction applied to circle centres, if any
    projection: "object | None" = None   # core.uncertainty.ProjectionUncertainty across the image, or None

    @property
    def rms_coordinate(self) -> float:
        """RMS over x and y residuals taken separately (the convention mrcal uses).

        For the same residuals it is the per-point value divided by sqrt(2).
        """
        return self.rms_point / np.sqrt(2.0)

    @property
    def fisheye(self) -> bool:
        return self.options.fisheye

    @property
    def distortion_names(self) -> list[str]:
        return lens.names(self.fisheye, len(self.distortion))

    @property
    def corner_total(self) -> int:
        return sum(v.corner_count for v in self.views)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


class CalibrationError(Exception):
    """Raised when the input cannot be calibrated. The message is shown to the user."""


def coverage_grid(
    detections: list[Detection], width: int, height: int
) -> tuple[np.ndarray, float]:
    """Corner counts on a coarse grid over the image, and the share of cells hit."""
    cols, rows = COVERAGE_GRID
    grid = np.zeros((rows, cols), np.int32)
    if width <= 0 or height <= 0:
        return grid, 0.0
    for det in detections:
        if det.count == 0:
            continue
        cx = np.clip((det.corners[:, 0] / width * cols).astype(int), 0, cols - 1)
        cy = np.clip((det.corners[:, 1] / height * rows).astype(int), 0, rows - 1)
        np.add.at(grid, (cy, cx), 1)
    return grid, float(np.count_nonzero(grid)) / grid.size


def calibrate(
    detections: list[Detection],
    target: Target,
    options: ModelOptions | None = None,
    excluded: set[int] | None = None,
) -> CalibrationResult:
    """Calibrate from detections.

    Views without enough corners are skipped; views whose index is in
    `excluded` are left out on purpose. Both are reported in the result.
    """
    options = options or ModelOptions()
    excluded = set(excluded or ())
    object_all = target.object_points()

    used: list[int] = []
    skipped: list[int] = []
    obj_pts, img_pts = [], []
    size: tuple[int, int] | None = None
    for i, det in enumerate(detections):
        if i in excluded:
            continue
        if not det.usable:
            skipped.append(i)
            continue
        if size is None:
            size = (det.width, det.height)
        elif size != (det.width, det.height):
            raise CalibrationError(
                "The images have different sizes. All images must come from the "
                f"same camera at the same resolution ({size[0]} x {size[1]} expected, "
                f"{det.width} x {det.height} found in {det.path})."
            )
        used.append(i)
        obj_pts.append(object_all[det.ids].reshape(-1, 1, 3).astype(np.float32))
        img_pts.append(det.corners.reshape(-1, 1, 2).astype(np.float32))

    if len(used) < MIN_VIEWS:
        raise CalibrationError(
            f"Only {len(used)} usable images. At least {MIN_VIEWS} are needed, "
            "and 15 to 30 from different angles give a much better result."
        )

    rms, K, dist, rvecs, tvecs = _solve(obj_pts, img_pts, size, options)

    # Circle grids: the detector gives the centre of each ellipse, not the image of the circle's
    # centre. With a first result in hand the difference can be computed and removed, then solved
    # again; the second round changes so little that two are enough.
    circle_shift = None
    if target.is_circles and target.marker_mm:
        from .circles import centre_bias

        raw = [p.reshape(-1, 2).astype(np.float64) for p in img_pts]
        for _round in range(2):
            shifts = [centre_bias(o.reshape(-1, 3), target.circle_diameter_mm / 2, r, t, K, dist, options.fisheye)
                      for o, r, t in zip(obj_pts, rvecs, tvecs)]
            img_pts = [(p - s).reshape(-1, 1, 2).astype(np.float32) for p, s in zip(raw, shifts)]
            rms, K, dist, rvecs, tvecs = _solve(obj_pts, img_pts, size, options)
        circle_shift = float(np.mean(np.linalg.norm(np.concatenate(shifts), axis=1)))

    views: list[ViewResult] = []
    total_sq, total_n = 0.0, 0
    for i, obj, img, rvec, tvec in zip(used, obj_pts, img_pts, rvecs, tvecs):
        proj = lens.project(obj, rvec, tvec, K, dist, options.fisheye)
        res = proj - img.reshape(-1, 2).astype(np.float64)
        sq = np.sum(res**2, axis=1)
        total_sq += float(sq.sum())
        total_n += len(sq)
        views.append(
            ViewResult(
                index=i,
                path=detections[i].path,
                corner_count=len(sq),
                rms_point=float(np.sqrt(sq.mean())),
                max_error=float(np.sqrt(sq.max())),
                rvec=np.asarray(rvec, np.float64).reshape(3),
                tvec=np.asarray(tvec, np.float64).reshape(3),
                residuals=res,
            )
        )
    rms_point = float(np.sqrt(total_sq / total_n))
    # OpenCV's own figure should agree with ours; keep ours, it is the one we can explain.
    if abs(rms_point - float(rms)) > 1e-3 * max(1.0, rms_point):
        raise CalibrationError(
            "Internal check failed: the reprojection error computed by Plumbline "
            f"({rms_point:.6f}) differs from OpenCV's ({float(rms):.6f})."
        )

    from .uncertainty import parameter_uncertainty, projection_uncertainty  # here to avoid a circular import

    uncertainty = parameter_uncertainty(
        K, dist, options, [(obj, v.rvec, v.tvec, v.residuals) for obj, v in zip(obj_pts, views)]
    )
    projection = None
    if uncertainty is not None:
        try:
            projection = projection_uncertainty(K, dist, options, uncertainty, size[0], size[1])
        except (cv2.error, np.linalg.LinAlgError, ValueError):
            projection = None  # the parameters still stand; only the map is left out
    result = CalibrationResult(
        circle_shift_px=circle_shift,
        uncertainty=uncertainty,
        projection=projection,
        image_width=size[0],
        image_height=size[1],
        camera_matrix=np.asarray(K, np.float64),
        distortion=dist,
        rms_point=rms_point,
        views=views,
        options=options,
        target=target,
        excluded=sorted(excluded),
        skipped=skipped,
    )
    from .points import point_statistics

    result.points = point_statistics(result, detections)
    return result


def _solve(obj_pts, img_pts, size, options: ModelOptions):
    """One run of OpenCV's solver. Returns rms, K, distortion (trimmed), rvecs, tvecs."""
    try:
        if options.fisheye:
            rms, K, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                [o.reshape(1, -1, 3).astype(np.float64) for o in obj_pts],
                [p.reshape(1, -1, 2).astype(np.float64) for p in img_pts],
                size, np.zeros((3, 3)), np.zeros((4, 1)),
                flags=cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
            )
        else:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_pts, img_pts, size, None, None, flags=options.flags()
            )
    except cv2.error as exc:  # pragma: no cover - depends on OpenCV internals
        hint = (" The fisheye model needs views that reach well into the corners of the image."
                if options.fisheye else "")
        raise CalibrationError(f"OpenCV could not solve this set of images.{hint} ({exc})") from exc
    return rms, K, _trim_distortion(np.asarray(dist, np.float64).reshape(-1), options), rvecs, tvecs


def _trim_distortion(dist: np.ndarray, options: ModelOptions) -> np.ndarray:
    """Cut the coefficient vector to the length the chosen model needs.

    OpenCV versions differ in how many trailing zeros they return. The
    result always has 5, 8 or 12 entries: k1 k2 p1 p2 k3 [k4 k5 k6 [s1 s2 s3 s4]];
    the fisheye model has 4.
    """
    if options.fisheye:
        want = 4
    else:
        want = 12 if options.thin_prism else 8 if options.rational else 5
    if len(dist) < want:
        dist = np.concatenate([dist, np.zeros(want - len(dist))])
    if np.any(dist[want:] != 0):
        raise CalibrationError("OpenCV returned lens terms that were not asked for.")
    return dist[:want].copy()


def suggest_exclusions(result: CalibrationResult, factor: float = 2.5) -> list[int]:
    """Views whose error stands well above the rest (median-based)."""
    if len(result.views) <= MIN_VIEWS:
        return []
    errors = np.array([v.rms_point for v in result.views])
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median))) or 1e-9
    limit = max(median * factor, median + 6.0 * mad)
    flagged = [v.index for v in result.views if v.rms_point > limit]
    keep = len(result.views) - len(flagged)
    return flagged if keep >= MIN_VIEWS else []


def projection_difference(
    K1: np.ndarray, d1: np.ndarray, K2: np.ndarray, d2: np.ndarray,
    width: int, height: int, step: int = 40, fisheye: bool = False,
) -> tuple[float, float]:
    """Mean and max pixel difference between two models over the image.

    Rays are taken through model 1 at a grid of pixels and projected with
    model 2. No rotation is fitted between the two, so this is an upper
    bound on how differently the two models see the same scene.
    """
    xs = np.arange(step / 2, width, step, dtype=np.float64)
    ys = np.arange(step / 2, height, step, dtype=np.float64)
    px = np.array([(x, y) for y in ys for x in xs], np.float64)
    rays = lens.pixel_directions(px, K1, d1, fisheye)
    ok = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1e-6)
    proj = lens.project(rays[ok], np.zeros(3), np.zeros(3), K2, d2, fisheye)
    diff = np.linalg.norm(proj - px[ok], axis=1)
    return float(diff.mean()), float(diff.max())
