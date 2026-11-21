"""The core is checked against a camera whose true parameters are known."""
import cv2
import numpy as np
import pytest

from plumbline.core import export, lens, synth
from plumbline.core.calibrate import (
    CalibrationError, ModelOptions, calibrate, coverage_grid,
    projection_difference, suggest_exclusions,
)
from plumbline.core.detect import Detection
from plumbline.core.findings import error_outside_fit, model_sentence
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

    xml = tmp_path / "camera.xml"
    export.write_opencv_xml(result, xml)
    assert xml.read_text().lstrip().startswith("<?xml")
    fs = cv2.FileStorage(str(xml), cv2.FILE_STORAGE_READ)
    assert np.allclose(fs.getNode("camera_matrix").mat(), result.camera_matrix)
    assert np.allclose(fs.getNode("distortion_coefficients").mat().reshape(-1), result.distortion)
    assert fs.getNode("avg_reprojection_error").real() == pytest.approx(result.rms_point)
    errors = fs.getNode("per_view_reprojection_errors").mat().reshape(-1)
    assert np.allclose(errors, [v.rms_point for v in result.views])
    names = fs.getNode("image_files")
    assert names.size() == len(result.views) and names.at(0).string() == "view-00.png"
    fs.release()


def test_project_round_trip(tmp_path, charuco, charuco_set):
    _poses, _images, dets = charuco_set
    p = Project(target=charuco, detections=dets, image_paths=[d.path for d in dets], excluded={2})
    p.save(str(tmp_path / "bench.plumbline"))
    q = Project.load(p.path)
    assert q.target == charuco and q.excluded == {2} and q.name == "bench"
    a, b = calibrate(dets, charuco, excluded={2}), calibrate(q.detections, q.target, excluded=q.excluded)
    assert np.allclose(a.camera_matrix, b.camera_matrix, rtol=1e-6)


def test_boards_of_a_set_are_told_apart_by_their_marker_ids(camera):
    """Board 3 of a set continues the numbering. Only the matching settings find it."""
    first = Target(CHARUCO, 12, 9, 60.0, 45.0, "DICT_5X5_1000")
    third = Target(CHARUCO, 12, 9, 60.0, 45.0, "DICT_5X5_1000", first_marker_id=108)
    assert first.marker_count == 54
    assert list(np.asarray(third.charuco_board().getIds()).reshape(-1)[[0, -1]]) == [108, 161]
    r, t = synth.sample_poses(third, camera, 1, seed=5)[0]
    image = synth.render(third, camera, r, t, np.random.default_rng(5))
    from plumbline.core.detect import detect
    right, wrong = detect(image, third), detect(image, first)
    assert right.count > 40
    truth = synth.true_corners(third, camera, r, t)[right.ids]
    assert np.median(np.linalg.norm(right.corners - truth, axis=1)) < 0.2
    assert wrong.count == 0
    assert "IDs 108 to 161" in wrong.note and "First marker ID 108" in wrong.note
    with pytest.raises(ValueError, match="only has IDs"):
        Target(CHARUCO, 12, 9, 60.0, 45.0, "DICT_5X5_50")


def test_names_of_projects_and_exported_files():
    from plumbline.core.naming import default_name, export_files, safe_name

    assert default_name(1) == "Calibration-001" and default_name(42) == "Calibration-042"
    assert safe_name('  bench cam: left/right?  ') == "bench-cam-leftright"
    assert safe_name("...") == "Calibration" and safe_name("con") == "Calibration"
    assert safe_name("左相机 01") == "左相机-01"
    one, two = export_files("Calibration-001", 1), export_files("Calibration-001", 2)
    assert one["report"] == "Calibration-001-report.pdf"
    assert one["opencv_yaml"] == "Calibration-001-camera.yml"
    assert two["report"] == "Calibration-001-r2-report.pdf"
    assert two["ros"] == "Calibration-001-r2-camera_info.yaml"
    assert one["project"] == two["project"] == "Calibration-001.plumbline"
    assert one["opencv_xml"] == "Calibration-001-camera.xml"
    assert two["xlsx"] == "Calibration-001-r2-data.xlsx"
    assert one["maps"] == "Calibration-001-undistort-maps.npz" and two["example"] == "Calibration-001-r2-example.py"
    assert len(set(one.values()) | set(two.values())) == 17


def test_excel_workbook_holds_the_same_numbers(tmp_path, charuco, charuco_set):
    from openpyxl import load_workbook

    from plumbline.core.export_xlsx import write_xlsx

    _poses, _images, dets = charuco_set
    result = calibrate(dets, charuco, excluded={4})
    path = tmp_path / "data.xlsx"
    write_xlsx(result, dets, path, "bench-cam", 2)
    wb = load_workbook(path)
    assert wb.sheetnames == ["Summary", "Images", "Corners"]
    summary = {row[0]: row[1] for row in wb["Summary"].iter_rows(min_row=2, values_only=True) if row[0]}
    assert summary["Project"] == "bench-cam" and summary["Revision"] == 2
    assert summary["fx, px"] == pytest.approx(result.fx)
    assert summary["k1"] == pytest.approx(result.distortion[0])
    assert summary["RMS per point, px"] == pytest.approx(result.rms_point)
    images = list(wb["Images"].iter_rows(min_row=2, values_only=True))
    assert len(images) == len(dets)
    assert images[4][4] == "excluded" and images[4][3] is None and images[4][2] > 0
    assert images[0][2] == pytest.approx(result.views[0].rms_point)
    corners = list(wb["Corners"].iter_rows(min_row=2, values_only=True))
    assert len(corners) == result.corner_total
    total = sum(c[6] ** 2 for c in corners) / len(corners)
    assert total ** 0.5 == pytest.approx(result.rms_point, rel=1e-6)


def test_recalibration_due_dates():
    from datetime import date

    from plumbline.core.project import due_date, is_due

    assert due_date("2026-11-22", 6) == "2027-05-22"
    assert due_date("2026-08-31", 6) == "2027-02-28"
    assert due_date("2026-11-22", 24) == "2028-11-22"
    assert due_date("2026-11-22", 0) is None and due_date(None, 6) is None
    assert is_due("2027-05-22", date(2027, 5, 22)) and not is_due("2027-05-22", date(2027, 5, 21))
    assert not is_due(None)


def test_every_stock_board_is_a_valid_target_and_can_be_found(camera):
    from plumbline.core.catalog import find_board, same_geometry, stock_boards
    from plumbline.core.detect import detect

    boards = stock_boards()
    assert len(boards) >= 20
    assert len({b.part_number for b in boards}) == len(boards)
    for b in boards:
        assert b.part_number.startswith("CT-") and b.product
        assert find_board(b.part_number) is b
        assert b.target.describe().startswith(b.part_number + ": ")
    assert find_board("CT-NOPE") is None and find_board(None) is None
    # One of each kind is rendered and found again with the settings the part number gives.
    for pn in ("CT-CHA-P-400X300-D5", "CT-CHT-P-300X200-C"):
        t = find_board(pn).target
        r, tv = synth.sample_poses(t, camera, 1, seed=3)[0]
        found = detect(synth.render(t, camera, r, tv, np.random.default_rng(3)), t)
        assert found.count > 0.6 * t.corner_count
    a = find_board("CT-CHA-P-400X300-D5").target
    assert same_geometry(a, Target(CHARUCO, a.squares_x, a.squares_y, a.square_mm, a.marker_mm, a.dictionary))


def test_fisheye_lens_is_recovered(charuco, fisheye_camera, fisheye_set):
    """A wide lens rendered with the fisheye model comes back from the fisheye fit."""
    poses, _images, dets = fisheye_set
    cam = fisheye_camera
    assert sum(d.usable for d in dets) >= 20
    for det, (r, t) in zip(dets, poses):
        if det.usable:
            truth = synth.true_corners(charuco, cam, r, t)[det.ids]
            assert np.median(np.linalg.norm(truth - det.corners, axis=1)) < 0.3

    result = calibrate(dets, charuco, ModelOptions(fisheye=True))
    assert result.fisheye and len(result.distortion) == 4
    assert result.distortion_names == ["k1", "k2", "k3", "k4"]
    assert abs(result.fx - cam.K[0, 0]) / cam.K[0, 0] < 0.002
    assert abs(result.fy - cam.K[1, 1]) / cam.K[1, 1] < 0.002
    assert abs(result.cx - cam.K[0, 2]) < 1.0 and abs(result.cy - cam.K[1, 2]) < 1.0
    assert result.rms_point < 0.3
    mean, _ = projection_difference(result.camera_matrix, result.distortion, cam.K, cam.dist,
                                    cam.width, cam.height, fisheye=True)
    assert mean < 0.3

    # The standard model with its default terms cannot describe this lens; the fit says so.
    assert calibrate(dets, charuco).rms_point > 3 * result.rms_point

    # An image left out of the fit still gets an error figure under the fisheye result.
    left_out = calibrate(dets, charuco, ModelOptions(fisheye=True), excluded={2})
    assert error_outside_fit(left_out, dets[2]) < 0.5


def test_fisheye_exports_say_which_model(tmp_path, charuco, fisheye_set):
    import json
    import yaml

    _poses, _images, dets = fisheye_set
    result = calibrate(dets, charuco, ModelOptions(fisheye=True))
    assert model_sentence(result) == "OpenCV fisheye model, radial terms k1 to k4."

    yml = tmp_path / "camera.yml"
    export.write_opencv_yaml(result, yml)
    fs = cv2.FileStorage(str(yml), cv2.FILE_STORAGE_READ)
    assert fs.getNode("distortion_model").string() == "opencv_fisheye"
    assert fs.getNode("distortion_coefficients").mat().shape == (1, 4)
    fs.release()

    xml = tmp_path / "camera.xml"
    export.write_opencv_xml(result, xml)
    fs = cv2.FileStorage(str(xml), cv2.FILE_STORAGE_READ)
    assert fs.getNode("distortion_model").string() == "opencv_fisheye"
    fs.release()

    js = tmp_path / "camera.json"
    export.write_json(result, js)
    data = json.loads(js.read_text())
    assert data["model"] == "opencv_fisheye" and data["distortion_names"] == ["k1", "k2", "k3", "k4"]
    assert data["options"]["fisheye"] is True

    ros = yaml.safe_load(export.ros_camera_info(result, "wide"))
    assert ros["distortion_model"] == "equidistant" and len(ros["distortion_coefficients"]["data"]) == 4

    # The standard model is named too, so a reader never has to guess.
    standard = calibrate(dets, charuco, ModelOptions(rational=True))
    assert export.result_summary(standard)["model"] == "opencv"
    assert 'distortion_model: "opencv"' in export.opencv_yaml(standard)


def test_corrected_view_straightens_the_board(charuco, camera, charuco_set, fisheye_camera, fisheye_set):
    """After correction the corners of a flat board fit a plane-to-plane mapping with no lens terms."""
    for cam, (_poses, images, dets), options in (
        (camera, charuco_set, ModelOptions()),
        (fisheye_camera, fisheye_set, ModelOptions(fisheye=True)),
    ):
        result = calibrate(dets, charuco, options)
        det = max(dets, key=lambda d: d.count)
        img = images[dets.index(det)]
        out, moved, new_K = lens.corrected_view(img, det.corners, result.camera_matrix,
                                                result.distortion, result.fisheye)
        assert out.shape == img.shape and new_K.shape == (3, 3)
        board = charuco.object_points()[det.ids][:, :2].astype(np.float64)

        def misfit(points):
            H, _ = cv2.findHomography(board, points, 0)
            back = cv2.perspectiveTransform(board.reshape(-1, 1, 2), H).reshape(-1, 2)
            return float(np.sqrt(np.mean(np.sum((back - points) ** 2, axis=1))))

        assert misfit(moved) < 0.25
        assert misfit(det.corners.astype(np.float64)) > 2 * misfit(moved)


def test_old_project_options_still_load():
    assert ModelOptions.from_dict({"k3": True, "something_newer": 1}) == ModelOptions(k3=True)
    assert ModelOptions.from_dict({}).fisheye is False


def test_lookup_tables_and_example_script(tmp_path, charuco, charuco_set, fisheye_set):
    """The exported tables and the exported script both reproduce Plumbline's corrected image."""
    import subprocess
    import sys

    from plumbline.core.export_code import write_example, write_maps

    for label, (_poses, images, dets), options in (
        ("standard", charuco_set, ModelOptions()),
        ("wide", fisheye_set, ModelOptions(fisheye=True)),
    ):
        result = calibrate(dets, charuco, options)
        folder = tmp_path / label
        folder.mkdir()
        export.write_opencv_yaml(result, folder / "cam-camera.yml")
        write_maps(result, folder / "cam-undistort-maps.npz")
        write_example(result, folder / "cam-example.py", "cam-camera.yml", "cam-undistort-maps.npz")

        img = images[3]
        ours, _moved, new_K = lens.corrected_view(img, np.zeros((0, 2)), result.camera_matrix,
                                                  result.distortion, result.fisheye)

        data = np.load(folder / "cam-undistort-maps.npz")
        assert data["map_x"].shape == img.shape and data["map_x"].dtype == np.float32
        assert tuple(data["image_size"]) == (result.image_width, result.image_height)
        assert str(data["distortion_model"]) == export.model_name(result)
        assert np.allclose(data["new_camera_matrix"], new_K)
        assert np.array_equal(cv2.remap(img, data["map_x"], data["map_y"], cv2.INTER_LINEAR), ours)

        # The script runs on its own, as a user would run it, and writes the same picture.
        cv2.imwrite(str(folder / "shot.png"), img)
        run = subprocess.run([sys.executable, str(folder / "cam-example.py"), str(folder / "shot.png")],
                             capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr
        assert ("OpenCV fisheye" if result.fisheye else "OpenCV standard") in run.stdout
        theirs = cv2.imread(str(folder / "shot-corrected.png"), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(theirs, ours)

        # Its helpers agree with Plumbline's own projection.
        sys.path.insert(0, str(folder))
        try:
            sys.modules.pop("cam-example", None)
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"example_{label}", folder / "cam-example.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)
        K, dist, size, fisheye = mod.load_camera()
        assert fisheye == result.fisheye and size == (result.image_width, result.image_height)
        v = result.views[0]
        obj = charuco.object_points()
        assert np.allclose(mod.project(obj, v.rvec, v.tvec, K, dist, fisheye),
                           lens.project(obj, v.rvec, v.tvec, result.camera_matrix, result.distortion, fisheye),
                           atol=1e-6)
        mx, my, nk = mod.load_maps()
        assert np.array_equal(mx, data["map_x"]) and np.allclose(nk, new_K)

        # A wrong-sized image is refused with a plain message, not a stack trace.
        cv2.imwrite(str(folder / "small.png"), img[::2, ::2])
        bad = subprocess.run([sys.executable, str(folder / "cam-example.py"), str(folder / "small.png")],
                             capture_output=True, text=True, timeout=120)
        assert bad.returncode != 0 and "this calibration is for" in bad.stderr and "Traceback" not in bad.stderr
