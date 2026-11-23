"""Write calibration results in the formats other software reads."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .. import __version__
from .calibrate import CalibrationResult
from .lens import STANDARD_NAMES as DIST_NAMES  # kept under the old name for callers


def model_name(result: CalibrationResult) -> str:
    """Name written into every export so the reader knows which OpenCV functions apply."""
    return "opencv_fisheye" if result.fisheye else "opencv"


def _fmt(value: float) -> str:
    return repr(float(value))


def _row(values) -> str:
    return "[" + ", ".join(_fmt(v) for v in values) + "]"


def result_summary(result: CalibrationResult) -> dict:
    """Plain-data form of a result, used by the JSON export and the project file."""
    dist = result.distortion.tolist()
    return {
        "generator": f"Plumbline {__version__} by Calib.Tools",
        "opencv_version": result.opencv_version,
        "model": model_name(result),
        "image_width": result.image_width,
        "image_height": result.image_height,
        "camera_matrix": result.camera_matrix.tolist(),
        "distortion_coefficients": dist,
        "distortion_names": result.distortion_names,
        "uncertainty": result.uncertainty.to_dict() if result.uncertainty else None,
        "projection_uncertainty": result.projection.to_dict() if result.projection else None,
        "residual_pattern": result.pattern.to_dict() if result.pattern else None,
        # The board as estimated, only when the result was solved with it.
        "board": result.board.to_dict() if result.board is not None and result.options.release_board else None,
        "rms_per_point_px": result.rms_point,
        "rms_per_coordinate_px": result.rms_coordinate,
        "rms_note": (
            "rms_per_point_px is sqrt(mean(dx^2 + dy^2)), the value OpenCV reports. "
            "rms_per_coordinate_px counts x and y separately and equals it divided by sqrt(2)."
        ),
        "views_used": len(result.views),
        "corners_used": result.corner_total,
        "target": result.target.to_dict(),
        "options": result.options.to_dict(),
        "views": [
            {
                "file": Path(v.path).name,
                "corners": v.corner_count,
                "rms_per_point_px": v.rms_point,
                "max_error_px": v.max_error,
                "rvec": v.rvec.tolist(),
                "tvec_mm": v.tvec.tolist(),
            }
            for v in result.views
        ],
        "excluded_files": [],
    }


def write_json(result: CalibrationResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result_summary(result), indent=2), encoding="utf-8")


def opencv_yaml(result: CalibrationResult) -> str:
    """OpenCV FileStorage YAML, readable with cv::FileStorage and cv2.FileStorage."""
    K = result.camera_matrix.reshape(-1)
    d = result.distortion.reshape(-1)
    note = ("# Fisheye model: use these values with the cv::fisheye functions.\n"
            if result.fisheye else "")
    return (
        "%YAML:1.0\n---\n"
        f"# Written by Plumbline {__version__} by Calib.Tools, https://calib.tools\n"
        f"{note}"
        f'distortion_model: "{model_name(result)}"\n'
        f"image_width: {result.image_width}\n"
        f"image_height: {result.image_height}\n"
        "camera_matrix: !!opencv-matrix\n"
        "   rows: 3\n   cols: 3\n   dt: d\n"
        f"   data: {_row(K)}\n"
        "distortion_coefficients: !!opencv-matrix\n"
        f"   rows: 1\n   cols: {len(d)}\n   dt: d\n"
        f"   data: {_row(d)}\n"
        f"avg_reprojection_error: {_fmt(result.rms_point)}\n"
    )


def write_opencv_yaml(result: CalibrationResult, path: str | Path) -> None:
    Path(path).write_text(opencv_yaml(result), encoding="utf-8")


def ros_distortion_model(result: CalibrationResult) -> str | None:
    """ROS distortion model name for this result, or None if ROS has none for it."""
    n = len(result.distortion)
    if result.fisheye:
        return "equidistant"
    if n == 5:
        return "plumb_bob"
    if n == 8:
        return "rational_polynomial"
    return None


def ros_camera_info(result: CalibrationResult, camera_name: str = "camera") -> str:
    """camera_info YAML as used by ROS camera_calibration_parsers.

    Raises ValueError for lens models ROS has no standard name for (thin prism).
    """
    model = ros_distortion_model(result)
    if model is None:
        raise ValueError(
            "ROS camera_info has no standard distortion model for a result with "
            f"{len(result.distortion)} coefficients. Calibrate without the thin prism terms."
        )
    K = result.camera_matrix
    P = np.hstack([K, np.zeros((3, 1))])
    R = np.eye(3)
    d = result.distortion
    return (
        f"# Written by Plumbline {__version__} by Calib.Tools, https://calib.tools\n"
        f"image_width: {result.image_width}\n"
        f"image_height: {result.image_height}\n"
        f"camera_name: {camera_name}\n"
        "camera_matrix:\n  rows: 3\n  cols: 3\n"
        f"  data: {_row(K.reshape(-1))}\n"
        f"distortion_model: {model}\n"
        f"distortion_coefficients:\n  rows: 1\n  cols: {len(d)}\n"
        f"  data: {_row(d)}\n"
        "rectification_matrix:\n  rows: 3\n  cols: 3\n"
        f"  data: {_row(R.reshape(-1))}\n"
        "projection_matrix:\n  rows: 3\n  cols: 4\n"
        f"  data: {_row(P.reshape(-1))}\n"
    )


def write_ros_camera_info(result: CalibrationResult, path: str | Path, camera_name: str = "camera") -> None:
    Path(path).write_text(ros_camera_info(result, camera_name), encoding="utf-8")


def write_opencv_xml(result: CalibrationResult, path: str | Path) -> None:
    """OpenCV FileStorage XML: the parameters plus the per-image figures of the report.

    Written by OpenCV itself, so cv::FileStorage reads it back in C++ and Python.
    The key names match the YAML export; the report figures are extra keys.
    """
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_FORMAT_XML)
    try:
        fs.write("generator", f"Plumbline {__version__} by Calib.Tools, https://calib.tools")
        fs.write("opencv_version", result.opencv_version)
        fs.write("distortion_model", model_name(result))
        fs.write("image_width", int(result.image_width))
        fs.write("image_height", int(result.image_height))
        fs.write("camera_matrix", np.asarray(result.camera_matrix, np.float64))
        fs.write("distortion_coefficients", np.asarray(result.distortion, np.float64).reshape(1, -1))
        fs.write("avg_reprojection_error", float(result.rms_point))
        fs.write("rms_per_coordinate", float(result.rms_coordinate))
        fs.write("target", result.target.describe())
        fs.write("views_used", len(result.views))
        fs.write("corners_used", int(result.corner_total))
        fs.startWriteStruct("image_files", cv2.FileNode_SEQ)
        for v in result.views:
            fs.write("", Path(v.path).name)
        fs.endWriteStruct()
        fs.write("per_view_corners", np.array([[v.corner_count] for v in result.views], np.int32))
        fs.write("per_view_reprojection_errors", np.array([[v.rms_point] for v in result.views], np.float64))
        fs.write("per_view_max_errors", np.array([[v.max_error] for v in result.views], np.float64))
        fs.write("rotation_vectors", np.array([v.rvec for v in result.views], np.float64))
        fs.write("translation_vectors_mm", np.array([v.tvec for v in result.views], np.float64))
    finally:
        fs.release()


# ------------------------------------------------------------- two cameras
def rig_summary(rig, files: tuple[str, str] | None = None, check=None) -> dict:
    """How camera 2 sits relative to camera 1, with both cameras' own parameters alongside."""
    from .rectify import rectify   # here, not at the top: rectify builds on rig, which builds on this package

    try:
        rectification = rectify(rig, tables=False)
    except cv2.error:
        rectification = None
    out = {
        "generator": f"Plumbline {__version__} by Calib.Tools, https://calib.tools",
        "convention": "x2 = rotation @ x1 + translation_mm: a point in camera 1 coordinates, in camera 2 coordinates. "
                      "Camera axes as in OpenCV: x to the right, y down, z the way the camera looks.",
        **rig.to_dict(),
        "std": {"rotation_vector_rad": [float(v) for v in rig.std[:3]],
                "translation_mm": [float(v) for v in rig.std[3:]],
                "note": "One standard deviation, from the scatter of the detected points and from how well "
                        "each camera's own parameters are known."},
        "pairs": [{"pair": p.index + 1, "rms_point": p.rms_point, "points": p.points} for p in rig.pairs],
        "rectification": rectification.to_dict() if rectification is not None else None,
        "camera_1": result_summary(rig.first),
        "camera_2": result_summary(rig.second),
    }
    if files:
        out["camera_files"] = {"camera_1": files[0], "camera_2": files[1]}
    if check is not None:
        out["check_with_other_images"] = {
            "note": "Pairs of images that took no part in the calibration. The two cameras measure lengths between "
                    "points of the target; these are compared with the target's own dimensions. measured_over_true "
                    "above 1 means lengths come out too long.",
            **check.to_dict(),
        }
    return out


def write_rig_json(rig, path: str | Path, files: tuple[str, str] | None = None, check=None) -> None:
    Path(path).write_text(json.dumps(rig_summary(rig, files, check), indent=2), encoding="utf-8")


def write_rig_yaml(rig, path: str | Path) -> None:
    """OpenCV FileStorage YAML with both cameras and R, T, as cv2.stereoRectify and cv::fisheye take them."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.writeComment(f"Written by Plumbline {__version__} by Calib.Tools, https://calib.tools")
        fs.writeComment("R, T: x2 = R * x1 + T, mm. K1, D1 belong to camera 1; K2, D2 to camera 2.")
        for n, cam in ((1, rig.first), (2, rig.second)):
            fs.write(f"distortion_model_{n}", model_name(cam))
            fs.write(f"image_width_{n}", int(cam.image_width))
            fs.write(f"image_height_{n}", int(cam.image_height))
            fs.write(f"K{n}", np.asarray(cam.camera_matrix, np.float64))
            fs.write(f"D{n}", np.asarray(cam.distortion, np.float64).reshape(1, -1))
        fs.write("R", np.asarray(rig.rotation, np.float64))
        fs.write("T", np.asarray(rig.translation, np.float64).reshape(3, 1))
        from .rectify import rectify

        try:
            r = rectify(rig, tables=False)
            fs.writeComment("Row alignment (stereoRectify): R1, R2, P1, P2, Q. The lenses are not in these; use "
                            "the lookup tables, or each camera's own model with R1 / P1 and R2 / P2.")
            for key in ("R1", "R2", "P1", "P2", "Q"):
                fs.write(key, np.asarray(getattr(r, key), np.float64))
        except cv2.error:
            pass
        fs.write("rms_point", float(rig.rms_point))
        fs.write("pairs_used", int(len(rig.pairs)))
    finally:
        fs.release()


def group_summary(group, files: list[str] | None = None) -> dict:
    """Where each camera sits relative to camera 1, with every camera's own parameters alongside."""
    out = {
        "generator": f"Plumbline {__version__} by Calib.Tools, https://calib.tools",
        "convention": "x = rotation @ x1 + translation_mm: a point in camera 1 coordinates, in that camera's "
                      "coordinates. Camera axes as in OpenCV: x to the right, y down, z the way the camera looks.",
        "camera_count": len(group.cameras),
        **group.to_dict(),
        "std_note": "One standard deviation, from the scatter of the detected points and from how well each "
                    "camera's own parameters are known.",
        "sets": [{"set": s.index + 1, "cameras": [c + 1 for c in s.cameras], "rms_point": s.rms_point,
                  "points": s.points} for s in group.sets],
    }
    for c, cam in enumerate(group.cameras):
        out[f"camera_{c + 1}"] = result_summary(cam)
    if files:
        out["camera_files"] = {f"camera_{c + 1}": f for c, f in enumerate(files)}
    return out


def write_group_json(group, path: str | Path, files: list[str] | None = None) -> None:
    Path(path).write_text(json.dumps(group_summary(group, files), indent=2), encoding="utf-8")


def write_group_yaml(group, path: str | Path) -> None:
    """OpenCV FileStorage YAML with every camera and, for cameras 2 and up, R and T from camera 1."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.writeComment(f"Written by Plumbline {__version__} by Calib.Tools, https://calib.tools")
        fs.writeComment("Rn, Tn: xn = Rn * x1 + Tn, mm. Kn, Dn belong to camera n.")
        fs.write("camera_count", int(len(group.cameras)))
        for n, cam in enumerate(group.cameras, 1):
            fs.write(f"distortion_model_{n}", model_name(cam))
            fs.write(f"image_width_{n}", int(cam.image_width))
            fs.write(f"image_height_{n}", int(cam.image_height))
            fs.write(f"K{n}", np.asarray(cam.camera_matrix, np.float64))
            fs.write(f"D{n}", np.asarray(cam.distortion, np.float64).reshape(1, -1))
        for p in group.placed:
            fs.write(f"R{p.camera + 1}", np.asarray(p.rotation, np.float64))
            fs.write(f"T{p.camera + 1}", np.asarray(p.translation, np.float64).reshape(3, 1))
        fs.write("rms_point", float(group.rms_point))
        fs.write("sets_used", int(len(group.sets)))
    finally:
        fs.release()
