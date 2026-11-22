"""Plain-language reading of a calibration result."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import lens
from .calibrate import CalibrationResult, suggest_exclusions
from .detect import Detection

GOOD_RMS = 0.5  # px per point; above this the result deserves a second look
LOOSE_CENTRE_SHARE = 0.002  # of the image width; beyond it the principal point is poorly determined
LOOSE_FOCAL_PERCENT = 0.2  # above this share of the focal length, more or better views are advised
PASS, CHECK, NOTE = "PASS", "CHECK", "NOTE"


def findings(result: CalibrationResult, coverage_share: float) -> list[tuple[str, str]]:
    """(tag, sentence) pairs, most important first."""
    r = result
    word = r.target.point_word.capitalize()
    out: list[tuple[str, str]] = []
    if r.rms_point <= GOOD_RMS:
        out.append((PASS, f"Error is low ({r.rms_point:.2f} px)."))
    else:
        out.append((CHECK, f"Error is {r.rms_point:.2f} px. Look for blurred images, "
                           "glare on the target, or wrong target sizes."))
    flagged = suggest_exclusions(r)
    if flagged:
        by_index = {v.index: v for v in r.views}
        names = ", ".join(Path(by_index[i].path).stem for i in flagged[:3])
        out.append((CHECK, f"{names} stand well above the rest. Look at them; a lower error "
                           "after excluding an image does not by itself mean a better result."))
    if coverage_share < 0.7:
        out.append((CHECK, f"{word} cover {coverage_share * 100:.0f} % of the image. "
                           "Add views that reach the empty cells in the coverage map."))
    else:
        out.append((PASS, f"{word} cover {coverage_share * 100:.0f} % of the image."))
    u = r.uncertainty
    centre = max((u.std_of(k) or 0.0) for k in ("cx", "cy")) if u is not None else 0.0
    if centre > LOOSE_CENTRE_SHARE * r.image_width:
        out.append((CHECK, f"The image centre is only pinned down to ±{centre:.1f} px. The lens model may have "
                           "more terms than these images can tell apart; try it with fewer."))
    sd = u.std_of("fx") if u is not None else None
    if sd is not None:
        share = 100 * sd / r.fx
        if share > LOOSE_FOCAL_PERCENT:
            out.append((CHECK, f"Focal length is only pinned down to ±{sd:.1f} px ({share:.2f} %). "
                               "Add images with the target tilted more strongly and at several distances."))
        else:
            out.append((NOTE, f"Focal length ±{sd:.2f} px ({share:.3f} %), one standard deviation "
                              "from the scatter of the points."))
    if len(r.views) < 10:
        out.append((NOTE, f"Only {len(r.views)} images in use. 15 to 30 give a steadier result."))
    if r.excluded:
        n = len(r.excluded)
        out.append((NOTE, f"{n} image{'s are' if n != 1 else ' is'} excluded by the user."))
    return out


def error_outside_fit(result: CalibrationResult, det: Detection) -> float | None:
    """RMS error an image would have under this result, for images that took no part in the fit."""
    if not det.usable:
        return None
    obj = result.target.object_points()[det.ids].astype(np.float64)
    K, dist, fisheye = result.camera_matrix, result.distortion, result.fisheye
    pose = lens.solve_pose(obj, det.corners, K, dist, fisheye)
    if pose is None:
        return None
    proj = lens.project(obj, pose[0], pose[1], K, dist, fisheye)
    return float(np.sqrt(np.mean(np.sum((proj - det.corners) ** 2, axis=1))))


def model_sentence(result: CalibrationResult) -> str:
    o = result.options
    if o.fisheye:
        return "OpenCV fisheye model, radial terms k1 to k4."
    parts = ["OpenCV standard model"]
    parts.append("k3 estimated" if o.k3 else "k3 fixed at 0")
    if not o.tangential:
        parts.append("no tangential terms")
    if o.rational:
        parts.append("rational terms k4 to k6")
    if o.thin_prism:
        parts.append("thin prism terms s1 to s4")
    return ", ".join(parts) + "."
