"""The core is checked against a camera whose true parameters are known."""
import cv2
import numpy as np
import pytest

from plumbline.core import export, synth
from plumbline.core.calibrate import (
    CalibrationError, ModelOptions, calibrate, coverage_grid,
    projection_difference, suggest_exclusions,
)
from plumbline.core.detect import Detection
from plumbline.core.project import Project
from plumbline.core.targets import CHARUCO, Target


def test_object_points_match_opencv(charuco):
    ours = charuco.object_points()
    theirs = np.asarray(charuco.charuco_board().getChessboardCorners(), np.float32)
    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs, atol=1e-5)


def test_target_validation():
    with pytest.raises(ValueError):
        Target(CHARUCO, 5, 7, 30.0, 31.0, "DICT_4X4_50")
    with pytest.raises(ValueError):
        Target(CHARUCO, 5, 7, 30.0, 22.0, "DICT_NOPE")


def test_charuco_corners_land_where_they_should(charuco, camera, charuco_set):
    poses, _images, dets = charuco_set
    errors = []
    for (r, t), det in zip(poses, dets):
        if det.count == 0:
            continue
        truth = synth.true_corners(charuco, camera, r, t)[det.ids]
        errors.append(np.linalg.norm(det.corners - truth, axis=1))
    errors = np.concatenate(errors)
    assert len(errors) > 0.6 * len(dets) * charuco.corner_count
    assert np.median(errors) < 0.15
    assert np.percentile(errors, 95) < 0.6


def test_charuco_calibration_recovers_the_camera(charuco, camera, charuco_set):
    _poses, _images, dets = charuco_set
    result = calibrate(dets, charuco)
    assert len(result.views) >= 18
    assert result.rms_point < 0.5
    assert abs(result.fx - 1500) / 1500 < 0.005
    assert abs(result.fy - 1500) / 1500 < 0.005
    assert abs(result.cx - 640) < 3
    assert abs(result.cy - 480) < 3
    mean, worst = projection_difference(
        camera.K, camera.dist, result.camera_matrix, result.distortion,
        camera.width, camera.height,
    )
    assert mean < 0.6
    assert worst < 2.0
    assert result.rms_coordinate == pytest.approx(result.rms_point / np.sqrt(2))


def test_checkerboard_calibration_recovers_the_camera(checkerboard, camera, checker_set):
    _poses, _images, dets = checker_set
    assert sum(d.usable for d in dets) >= 12
    result = calibrate(dets, checkerboard)
    assert result.rms_point < 0.5
    assert abs(result.fx - 1500) / 1500 < 0.006
    mean, _ = projection_difference(
        camera.K, camera.dist, result.camera_matrix, result.distortion,
        camera.width, camera.height,
    )
    assert mean < 0.8


def test_a_bad_view_is_flagged_and_excluding_it_helps(charuco, charuco_set):
    _poses, _images, dets = charuco_set
    usable = [i for i, d in enumerate(dets) if d.usable]
    victim = usable[3]
    rng = np.random.default_rng(3)
    spoiled = list(dets)
    d = dets[victim]
    spoiled[victim] = Detection(d.path, d.width, d.height, d.ids.copy(),
                                d.corners + rng.normal(0, 4.0, d.corners.shape).astype(np.float32))
    first = calibrate(spoiled, charuco)
    assert victim in suggest_exclusions(first)
    second = calibrate(spoiled, charuco, excluded={victim})
    assert second.rms_point < first.rms_point
    assert victim in second.excluded
    assert all(v.index != victim for v in second.views)


def test_too_few_views_gives_a_readable_error(charuco, charuco_set):
    _poses, _images, dets = charuco_set
    with pytest.raises(CalibrationError, match="usable images"):
        calibrate(dets[:2], charuco)


def test_mixed_image_sizes_are_refused(charuco, charuco_set):
    _poses, _images, dets = charuco_set
    odd = Detection("other.png", 640, 480, dets[0].ids, dets[0].corners)
    with pytest.raises(CalibrationError, match="different sizes"):
        calibrate([odd] + [d for d in dets if d.usable], charuco)


def test_model_options_change_the_coefficient_count(charuco, charuco_set):
    _poses, _images, dets = charuco_set
    assert len(calibrate(dets, charuco).distortion) == 5
    assert calibrate(dets, charuco).distortion[4] == 0.0
    assert len(calibrate(dets, charuco, ModelOptions(rational=True)).distortion) == 8
    assert len(calibrate(dets, charuco, ModelOptions(thin_prism=True)).distortion) == 12


def test_coverage(charuco_set, camera):
    _poses, _images, dets = charuco_set
    grid, share = coverage_grid(dets, camera.width, camera.height)
    assert grid.sum() == sum(d.count for d in dets)
    assert 0.5 < share <= 1.0


def test_exports_round_trip(tmp_path, charuco, charuco_set):
    _poses, _images, dets = charuco_set
    result = calibrate(dets, charuco)

    yml = tmp_path / "camera.yml"
    export.write_opencv_yaml(result, yml)
    fs = cv2.FileStorage(str(yml), cv2.FILE_STORAGE_READ)
    assert np.allclose(fs.getNode("camera_matrix").mat(), result.camera_matrix)
    assert np.allclose(fs.getNode("distortion_coefficients").mat().reshape(-1), result.distortion)
    assert int(fs.getNode("image_width").real()) == result.image_width
    fs.release()

    import json
    import yaml
    js = tmp_path / "camera.json"
    export.write_json(result, js)
    data = json.loads(js.read_text())
    assert np.allclose(data["camera_matrix"], result.camera_matrix)
    assert data["rms_per_coordinate_px"] == pytest.approx(data["rms_per_point_px"] / np.sqrt(2))

    ros = yaml.safe_load(export.ros_camera_info(result, "bench"))
    assert ros["distortion_model"] == "plumb_bob"
    assert np.allclose(ros["camera_matrix"]["data"], result.camera_matrix.reshape(-1))
    assert len(ros["projection_matrix"]["data"]) == 12

    prism = calibrate(dets, charuco, ModelOptions(thin_prism=True))
    with pytest.raises(ValueError):
        export.ros_camera_info(prism)


def test_project_round_trip(tmp_path, charuco, charuco_set):
    _poses, _images, dets = charuco_set
    p = Project(target=charuco, detections=dets, image_paths=[d.path for d in dets], excluded={2})
    p.save(str(tmp_path / "bench.plumbline"))
    q = Project.load(p.path)
    assert q.target == charuco and q.excluded == {2} and q.name == "bench"
    a, b = calibrate(dets, charuco, excluded={2}), calibrate(q.detections, q.target, excluded=q.excluded)
    assert np.allclose(a.camera_matrix, b.camera_matrix, rtol=1e-6)
