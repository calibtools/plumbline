"""Plain-language reading of a calibration result."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import lens
from .calibrate import CalibrationResult, suggest_exclusions
from .detect import Detection

GOOD_RMS = 0.5  # px per point; above this the result deserves a second look
LOOSE_CENTRE_SHARE = 0.002  # of the image width; beyond it the principal point is poorly determined
LOOSE_EDGE_PX = 0.5  # projection uncertainty anywhere in the image above which more views are advised
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
    board = r.board
    misfit = r.pattern is not None and r.pattern.has_pattern
    if board is not None and r.options.release_board:
        shape = f"{board.out_of_flat_mm:.2f} mm out of flat, up to {board.in_plane_mm:.2f} mm off within its plane"
        if board.clearly_better:
            out.append((NOTE, f"The shape of the board was estimated with the camera: {shape}. "
                              f"With the board as entered the error would be {board.rms_nominal:.2f} px."))
        else:
            out.append((NOTE, f"The shape of the board was estimated with the camera and came out close to the "
                              f"board as entered ({shape}). Estimating it was not needed here and makes the "
                              "result a little less certain; consider switching it off."))
        if not board.settled:
            out.append((CHECK, "The estimate of the board's shape was still moving when it was stopped. "
                               "Treat this result with care; more images of the whole board help."))
    elif board is not None and board.clearly_better and board.whole_board and not misfit:
        flat = (f"about {board.out_of_flat_mm:.1f} mm out of flat" if board.out_of_flat_mm >= 0.1
                else "not quite the shape entered")
        out.append((CHECK, f"These images fit clearly better if the board is {flat} "
                           f"(error {board.rms_released:.2f} instead of {board.rms_nominal:.2f} px). "
                           "A result that takes the board to be flat and to size is then off. "
                           "Check the board and its measurements, or tick 'Estimate the shape of the board' "
                           "on the Model tab."))
    pattern = r.pattern
    if pattern is not None and pattern.has_pattern:
        if r.options.fisheye:
            advice = "The fisheye model does not follow this lens; try the standard model with more terms."
        elif r.options.k3 and r.options.rational:
            advice = "If the lens is very wide, try the fisheye model."
        else:
            advice = "Add lens terms (k3, then the rational terms), or the fisheye model for a very wide lens."
        out.append((CHECK, f"What is left over after the fit is not just noise: in parts of the image it leans "
                           f"one way by up to {pattern.largest_px:.2f} px. {advice} See the Residuals map."))
    stats = r.points
    if stats is not None:
        from .points import trusted_suspects

        bad = trusted_suspects(r)
        if bad:
            obj = r.target.object_points()
            first = bad[0]
            x, y = obj[int(stats.ids[first])][:2]
            size = float(np.linalg.norm(stats.offset_mm[first]))
            more = f" {len(bad) - 1} more like it." if len(bad) > 1 else ""
            out.append((CHECK, f"One point of the board is off the same way in every image: about {size:.2f} mm, "
                               f"at {x:g} mm across and {y:g} mm down from the first point. "
                               f"Look at the board there; it may be damaged or not flat.{more}"))
    if r.target.is_circles:
        if r.circle_shift_px is not None:
            out.append((NOTE, f"Circle centres were corrected for perspective, {r.circle_shift_px:.2f} px on average."))
        else:
            out.append((NOTE, "Circle diameter not entered: centres are used as detected, without the "
                              "correction for perspective. Enter it on the Target tab."))
    pu = r.projection
    if pu is not None and pu.worst >= LOOSE_EDGE_PX:
        out.append((CHECK, f"Towards the edge of the image the result is only good to ±{pu.worst:.1f} px "
                           f"(±{pu.median:.2f} px over most of it). Add views that put the target there; "
                           "see the Uncertainty map."))
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
    obj = result.board_points()[det.ids]
    K, dist, fisheye = result.camera_matrix, result.distortion, result.fisheye
    pose = lens.solve_pose(obj, det.corners, K, dist, fisheye)
    if pose is None:
        return None
    seen = det.corners.astype(np.float64)
    if result.target.is_circles and result.target.marker_mm:
        # Same treatment as the images in the fit: take off the perspective shift of the centres.
        from .circles import centre_bias

        seen = seen - centre_bias(obj, result.target.circle_diameter_mm / 2, pose[0], pose[1], K, dist, fisheye)
        pose = lens.solve_pose(obj, seen, K, dist, fisheye) or pose
    proj = lens.project(obj, pose[0], pose[1], K, dist, fisheye)
    return float(np.sqrt(np.mean(np.sum((proj - seen) ** 2, axis=1))))


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


def rig_findings(rig) -> list[tuple[str, str]]:
    """What to know about how two cameras sit to each other, before each camera's own findings."""
    out = []
    n = len(rig.pairs)
    errors = sorted(p.rms_point for p in rig.pairs)
    middle = errors[n // 2]
    if n < 10:
        out.append((CHECK, f"Only {n} pairs show the target in both cameras. With 15 or more, from different "
                           "angles and distances, the position of camera 2 is pinned down much better."))
    else:
        out.append((PASS, f"{n} pairs show the target in both cameras."))
    worst = max(rig.pairs, key=lambda p: p.rms_point)
    if n >= 4 and worst.rms_point > 3 * middle and worst.rms_point > 0.3:
        out.append((CHECK, f"Pair {worst.index + 1:02d} fits much worse than the others ({worst.rms_point:.2f} px "
                           f"against a typical {middle:.2f} px). The target or a camera may have moved between "
                           "the two exposures. Try the result without this pair."))
    if rig.apart_mm > 0 and rig.apart_std_mm == rig.apart_std_mm:
        share = rig.apart_std_mm / rig.apart_mm
        text = (f"Distance between the cameras {rig.apart_mm:.2f} mm ±{rig.apart_std_mm:.2f} mm "
                f"({share * 100:.2f} %), angle ±{rig.turned_std_deg:.3f}°, one standard deviation.")
        if share > 0.005:
            out.append((CHECK, text + " Depth measured with this pair of cameras is off by about the same share. "
                                      "Add pairs with the target close to the cameras and tilted."))
        else:
            out.append((NOTE, text))
    if rig.first.options.release_board and rig.second.options.release_board:
        out.append((NOTE, "The shape of the board was estimated by each camera on its own and the mean of the two "
                          "was used for both. How well that shape is known is not part of the ± values."))
    if rig.unpaired:
        k = len(rig.unpaired)
        out.append((NOTE, f"{k} pair{' takes' if k == 1 else 's take'} no part in the position of camera 2: one of "
                          "the two images has no usable detection, or the pair was excluded."))
    return out


def group_findings(group) -> list[tuple[str, str]]:
    """What to know about how three or more cameras sit to each other, before each camera's own findings."""
    out = []
    n = len(group.sets)
    errors = sorted(s.rms_point for s in group.sets)
    middle = errors[n // 2]
    thin = [p for p in group.placed if p.sets < 10]
    if thin:
        names = ", ".join(f"camera {p.camera + 1} ({p.sets})" for p in thin)
        out.append((CHECK, f"Few sets show the target in {names} and in another camera. With 15 or more, from "
                           "different angles and distances, its position is pinned down much better."))
    else:
        out.append((PASS, f"{n} sets show the target in two or more cameras."))
    worst = max(group.sets, key=lambda s: s.rms_point)
    if n >= 4 and worst.rms_point > 3 * middle and worst.rms_point > 0.3:
        out.append((CHECK, f"Set {worst.index + 1:02d} fits much worse than the others ({worst.rms_point:.2f} px "
                           f"against a typical {middle:.2f} px). The target or a camera may have moved between "
                           "the exposures. Try the result without this set."))
    for p in group.placed:
        if not (p.apart_mm > 0 and p.apart_std_mm == p.apart_std_mm):
            continue
        share = p.apart_std_mm / p.apart_mm
        text = (f"Camera {p.camera + 1} is {p.apart_mm:.2f} mm from camera 1, ±{p.apart_std_mm:.2f} mm "
                f"({share * 100:.2f} %), angle ±{p.turned_std_deg:.3f}°, one standard deviation.")
        if share > 0.005:
            out.append((CHECK, text + " Add sets with the target close to both cameras and tilted."))
        else:
            out.append((NOTE, text))
    for p in group.placed:
        if p.through != 0:
            out.append((NOTE, f"Camera {p.camera + 1} was first placed through camera {p.through + 1}: too few sets "
                              "show the target in it and in camera 1."))
    if all(c.options.release_board for c in group.cameras):
        out.append((NOTE, "The shape of the board was estimated by each camera on its own and the mean was used "
                          "for all. How well that shape is known is not part of the ± values."))
    if group.unused:
        k = len(group.unused)
        out.append((NOTE, f"{k} set{' takes' if k == 1 else 's take'} no part in where the cameras sit: fewer than "
                          "two images have a usable detection, or the set was excluded."))
    return out
