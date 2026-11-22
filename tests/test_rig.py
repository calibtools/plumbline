"""Two cameras looking at the same board: where the second sits relative to the first."""
import cv2
import numpy as np
import pytest

from plumbline.core import lens, synth
from plumbline.core.calibrate import CalibrationError, ModelOptions
from plumbline.core.detect import Detection, detect
from plumbline.core.rig import calibrate_pair

TURN = np.deg2rad(4.0)
R_TRUE = cv2.Rodrigues(np.array([0.0, -TURN, 0.0]))[0]     # camera 2 turned inwards by 4 degrees
T_TRUE = np.array([-120.0, 2.0, 5.0])                      # and 120 mm to the right of camera 1


def second_pose(r, t):
    R1 = cv2.Rodrigues(np.asarray(r, np.float64))[0]
    return cv2.Rodrigues(R_TRUE @ R1)[0].ravel(), R_TRUE @ np.asarray(t, np.float64).ravel() + T_TRUE


def rendered_pairs(target, camera, count, seed):
    rng = np.random.default_rng(seed)
    first, second = [], []
    for i, (r, t) in enumerate(synth.sample_poses(target, camera, count, seed)):
        r2, t2 = second_pose(r, t)
        first.append(detect(synth.render(target, camera, r, t, rng), target, f"cam1-{i:02d}.png"))
        second.append(detect(synth.render(target, camera, r2, t2, rng), target, f"cam2-{i:02d}.png"))
    return first, second


def exact_pairs(target, camera, wide, count, seed, noise=0.08):
    """Exact projections plus pixel noise; points outside either image are dropped."""
    rng = np.random.default_rng(seed)
    board = target.object_points().astype(np.float64)
    first, second = [], []
    for i, (r, t) in enumerate(synth.sample_poses(target, camera, count, 7)):
        for store, (rr, tt), name in ((first, (r, t), "cam1"), (second, second_pose(r, t), "cam2")):
            p = lens.project(board, rr, tt, camera.K, camera.dist, wide)
            ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
            store.append(Detection(f"{name}-{i:02d}.png", camera.width, camera.height, np.nonzero(ok)[0].astype(np.int32),
                                   (p[ok] + rng.normal(0, noise, (int(ok.sum()), 2))).astype(np.float32)))
    return first, second


def off(rig):
    angle = np.rad2deg(np.linalg.norm(cv2.Rodrigues(rig.rotation @ R_TRUE.T)[0]))
    return float(np.linalg.norm(rig.translation - T_TRUE)), float(angle)


@pytest.mark.parametrize("wide", [False, True])
def test_two_rendered_cameras(wide, charuco, camera, fisheye_camera):
    cam = fisheye_camera if wide else camera
    first, second = rendered_pairs(charuco, cam, 16, 7)
    full = charuco.corner_count
    assert any(len(d.ids) < full for d in first + second)      # the board is partly out of view in some images
    rig = calibrate_pair(first, second, charuco, ModelOptions(fisheye=wide))
    shift, angle = off(rig)
    assert len(rig.pairs) == 16 and not rig.unpaired            # every pair counts, also with part of the board missing
    assert shift < 0.25 and angle < 0.03 and rig.rms_point < 0.2
    assert rig.apart_mm == pytest.approx(np.linalg.norm(T_TRUE), abs=0.2)
    assert rig.turned_deg == pytest.approx(4.0, abs=0.03)
    assert rig.position == pytest.approx(-R_TRUE.T @ T_TRUE, abs=0.3)   # camera 2 is to the right of camera 1
    assert rig.position[0] > 100
    assert 0 < rig.apart_std_mm < 0.3 and 0 < rig.turned_std_deg < 0.1
    d = rig.to_dict()
    assert d["apart_mm"] == rig.apart_mm and len(d["pairs_used"]) == 16

    if not wide:
        # Same data through OpenCV's own function, which only takes the points both cameras saw.
        O, A, B = [], [], []
        board = charuco.object_points().astype(np.float32)
        for a, b in zip(first, second):
            common = np.intersect1d(a.ids, b.ids)
            ia = {int(k): j for j, k in enumerate(a.ids)}
            ib = {int(k): j for j, k in enumerate(b.ids)}
            O.append(board[common])
            A.append(a.corners[[ia[int(k)] for k in common]])
            B.append(b.corners[[ib[int(k)] for k in common]])
        out = cv2.stereoCalibrate(O, A, B, rig.first.camera_matrix.copy(), rig.first.distortion.copy(),
                                  rig.second.camera_matrix.copy(), rig.second.distortion.copy(),
                                  (cam.width, cam.height), flags=cv2.CALIB_FIX_INTRINSIC)
        assert np.linalg.norm(out[6].ravel() - rig.translation) < 0.1
        assert np.rad2deg(np.linalg.norm(cv2.Rodrigues(out[5] @ rig.rotation.T)[0])) < 0.01


def test_a_pair_with_one_side_missing(charuco, camera):
    first, second = exact_pairs(charuco, camera, False, 12, 1)
    second[5] = Detection("cam2-05.png", camera.width, camera.height, note="No target found")
    rig = calibrate_pair(first, second, charuco)
    assert rig.unpaired == [5] and len(rig.pairs) == 11
    assert 5 in {v.index for v in rig.first.views} and 5 not in {v.index for v in rig.second.views}
    shift, angle = off(rig)
    assert shift < 0.5 and angle < 3 * rig.turned_std_deg + 0.02     # 11 pairs with 0.08 px noise

    rig = calibrate_pair(first, second, charuco, excluded={2})
    assert rig.unpaired == [2, 5]
    with pytest.raises(CalibrationError, match="same number"):
        calibrate_pair(first, second[:-1], charuco)
    empty = [Detection(d.path, d.width, d.height) for d in second]
    with pytest.raises(CalibrationError):
        calibrate_pair(first, empty, charuco)


def test_one_bad_pair_does_not_spoil_the_start(charuco, camera):
    first, second = exact_pairs(charuco, camera, False, 12, 2)
    # In one pair the board moved between the two exposures: camera 2 sees it 15 mm further left.
    bad = second[3]
    second[3] = Detection(bad.path, bad.width, bad.height, bad.ids, bad.corners + np.float32([18.0, 0.0]))
    rig = calibrate_pair(first, second, charuco)
    worst = max(rig.pairs, key=lambda p: p.rms_point)
    assert worst.index == 3 and worst.rms_point > 3 * np.median([p.rms_point for p in rig.pairs])
    clean = calibrate_pair(first, second, charuco, excluded={3})
    assert off(clean)[0] < 0.3 and clean.rms_point < 0.12


def test_findings_about_the_pair_of_cameras(charuco, camera):
    from plumbline.core.findings import CHECK, NOTE, PASS, rig_findings

    first, second = exact_pairs(charuco, camera, False, 12, 2)
    good = rig_findings(calibrate_pair(first, second, charuco))
    assert good[0] == (PASS, "12 pairs show the target in both cameras.")
    assert any(tag == NOTE and "Distance between the cameras 120." in t for tag, t in good)
    assert not any("fits much worse" in t for _tag, t in good)
    # The board moved between the two exposures of one pair.
    bad = second[3]
    second[3] = Detection(bad.path, bad.width, bad.height, bad.ids, bad.corners + np.float32([18.0, 0.0]))
    found = rig_findings(calibrate_pair(first, second, charuco))
    assert any(tag == CHECK and t.startswith("Pair 04 fits much worse") for tag, t in found)
    # Few pairs, and one left out.
    few = rig_findings(calibrate_pair(first[:6], second[:6], charuco, excluded={3}))
    assert few[0][0] == CHECK and few[0][1].startswith("Only 5 pairs")
    assert any(tag == NOTE and t.startswith("1 pair takes no part") for tag, t in few)


def test_circle_centres_are_corrected_in_both_cameras(camera):
    """The joint fit takes the points as each camera's own solve took them, perspective correction included."""
    from plumbline.core.targets import CIRCLES, Target

    board = Target(CIRCLES, 9, 7, 20.0, 12.0)
    first, second = rendered_pairs(board, camera, 30, 7)
    corrected = calibrate_pair(first, second, board)
    plain = calibrate_pair(first, second, Target(CIRCLES, 9, 7, 20.0, None))      # diameter not known: no correction
    assert len(corrected.pairs) >= 3 and corrected.first.circle_shift_px > 0.05 and plain.first.circle_shift_px is None
    assert off(corrected)[0] < 0.03 and off(corrected)[0] < off(plain)[0]


def test_an_estimated_board_shape_is_used_for_both_cameras(charuco, camera):
    from test_board import shaped

    from plumbline.core.findings import NOTE, rig_findings

    truth = shaped(charuco, sag_mm=1.0)
    rng = np.random.default_rng(5)
    first, second = [], []
    for i, (r, t) in enumerate(synth.sample_poses(charuco, camera, 24, 7)):
        for store, (rr, tt) in ((first, (r, t)), (second, second_pose(r, t))):
            p = lens.project(truth, rr, tt, camera.K, camera.dist, False)
            ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
            store.append(Detection(f"v{i}.png", camera.width, camera.height, np.nonzero(ok)[0].astype(np.int32),
                                   (p[ok] + rng.normal(0, 0.08, (int(ok.sum()), 2))).astype(np.float32)))
    flat = calibrate_pair(first, second, charuco)
    free = calibrate_pair(first, second, charuco, ModelOptions(release_board=True))
    # Taken as flat, a board that sags by 1 mm puts camera 2 about 2 mm and half a degree off.
    assert off(flat)[0] > 1.0 and off(flat)[1] > 0.2
    assert off(free)[0] < 0.6 and off(free)[1] < 0.1 and free.rms_point < 0.15
    # With the shape estimated, every stated ± holds: distance, angle, and each direction of the position.
    # The forward direction is the loose one; the distance between the cameras is known much better.
    true_place = -R_TRUE.T @ T_TRUE
    assert np.all(np.abs(free.position - true_place) < 3 * free.position_std)
    assert abs(free.apart_mm - np.linalg.norm(T_TRUE)) < 3 * free.apart_std_mm
    assert abs(free.turned_deg - 4.0) < 3 * free.turned_std_deg
    assert free.position_std[2] > 3 * free.position_std[0]
    # Taken as flat, the same ± is far too small: the board is wrong, and no ± can know that.
    assert abs(flat.apart_mm - np.linalg.norm(T_TRUE)) > 4 * flat.apart_std_mm
    assert any(tag == NOTE and "How well that shape is known is not part of the ± values" in t for tag, t in rig_findings(free))
    assert not any("shape of the board" in t for _tag, t in rig_findings(flat))


def test_stated_spread_matches_repeats(charuco, camera):
    """The ± on distance and angle against what repeated calibrations with new noise really do."""
    apart, turned, stated = [], [], []
    for seed in range(10):
        first, second = exact_pairs(charuco, camera, False, 10, 100 + seed)
        rig = calibrate_pair(first, second, charuco)
        apart.append(rig.apart_mm); turned.append(rig.turned_deg); stated.append((rig.apart_std_mm, rig.turned_std_deg))
    stated = np.median(np.array(stated), axis=0)
    print("apart: scatter", np.std(apart, ddof=1), "stated", stated[0], " turned: scatter", np.std(turned, ddof=1), "stated", stated[1])
    # Ten repeats pin the real scatter down to about a quarter, so the comparison is a coarse one.
    assert 0.5 < stated[0] / np.std(apart, ddof=1) < 2.0
    assert 0.5 < stated[1] / np.std(turned, ddof=1) < 2.0


def test_two_cameras_in_the_window(tmp_path, charuco, camera):
    pytest.importorskip("PySide6")
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app
    from plumbline.core.project import Project

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    rng = np.random.default_rng(7)
    one, two = [], []
    for i, (r, t) in enumerate(synth.sample_poses(charuco, camera, 8, 7)):
        for store, pose, name in ((one, (r, t), "cam1"), (two, second_pose(r, t), "cam2")):
            store.append(str(tmp_path / f"{name}-{i:02d}.png"))
            cv2.imwrite(store[-1], synth.render(charuco, camera, pose[0], pose[1], rng))

    def settle(bench):
        end, quiet = time.time() + 240, 0
        while time.time() < end and quiet <= 40:
            app.processEvents()
            time.sleep(0.02)
            quiet = 0 if bench._busy() else quiet + 1

    win = MainWindow()
    win.show()
    win.new_project()
    bench = win.bench
    bench._fill_fields(charuco)
    bench._target_changed()
    assert bench.cams_box.isHidden() and bench.view2.isHidden() and bench.rig_box.isHidden()
    bench.add_images(one)
    settle(bench)
    alone = bench.result
    assert alone is not None and bench.rig is None and bench.table.columnCount() == 3

    # A second camera: its images are added in any order and pair by file name.
    bench._toggle_second_camera()
    assert bench.project.second_camera and bench._cam == 1 and bench.result is None
    assert not bench.cams_box.isHidden() and not bench.view2.isHidden()
    assert bench.images_head.text() == "PAIRS" and bench.table.columnCount() == 4
    assert bench.table.item(0, 2).text() == "–" and bench.table.item(0, 3).text() == ""   # nothing to pair with yet
    bench.add_images(list(reversed(two[:5])))
    settle(bench)
    assert bench.rig is None and "camera 2 has 5" in bench.status.text()      # not yet one image each
    assert bench.table.item(6, 2).text() == "–" and bench.table.item(6, 3).text() == "no pair"
    bench.add_images(two[5:])
    settle(bench)
    assert [os.path.basename(p) for p in bench.project.second_image_paths] == [os.path.basename(p) for p in two]
    rig = bench.rig
    assert rig is not None and len(rig.pairs) == 8
    assert rig.apart_mm == pytest.approx(np.linalg.norm(T_TRUE), abs=0.5)
    assert bench.result is rig.second                                         # camera 2 is the one selected
    assert bench.rig_values["Apart"].text() == f"{rig.apart_mm:.2f} ± {rig.apart_std_mm:.2f} mm"
    assert bench.rig_values["Z (forward), mm"].text().startswith(f"{rig.position[2]:.2f} ± ")
    assert not any(v.text().endswith("± 0.00") for v in bench.rig_values.values())
    assert bench.rms_big.text() == f"{rig.rms_point:.3f} px"
    assert bench.steps[1]._status.text() == "2 cameras · 8 pairs"
    assert bench.table.item(0, 3).text() == f"{rig.pairs[0].rms_point:.3f}"
    assert bench.images_count.text() == "8 pairs · 8 used"
    bench._select_camera(0)
    assert bench.result is rig.first and bench.intr[1]["fx"].text() == f"{rig.first.fx:.2f}"

    # Leaving a pair out takes it from both cameras.
    bench._toggle(2)
    settle(bench)
    assert bench.rig.unpaired == [2] and bench.table.item(2, 3).text() == "out"
    assert 2 not in {v.index for v in bench.rig.first.views} | {v.index for v in bench.rig.second.views}
    bench._toggle(2)
    settle(bench)
    assert not bench.rig.unpaired

    # Export: every chosen file once per camera, and two files about how the cameras sit to each other.
    import json

    from plumbline.ui.export_dialog import ExportDialog

    assert bench.steps[4]._status.text() == "Files and report"
    rig = bench.rig
    name = bench.project.name
    dialog = ExportDialog(bench.project, bench.result, bench, rig=rig)
    assert dialog.file_names()["opencv_yaml"] == f"{name}-cam1-camera.yml"
    assert dialog.file_names()["rig_yaml"] == f"{name}-cameras.yml"
    assert "cam1 and again for cam2" in dialog.revision_note.text()
    out = tmp_path / "out"
    out.mkdir()
    bench.export_to(str(out), formats={"opencv_yaml", "json", "ros", "report", "xlsx", "maps"})
    expected = {f"{name}-cam{c}-{tail}" for c in (1, 2)
                for tail in ("camera.yml", "camera.json", "camera_info.yaml", "report.pdf", "data.xlsx",
                             "undistort-maps.npz")}
    expected |= {f"{name}-cameras.json", f"{name}-cameras.yml", f"{name}-cameras-report.pdf",
                 f"{name}-cameras-rectify-maps.npz", f"{name}.plumbline"}
    assert {p.name for p in out.iterdir()} == expected
    assert bench.status.text().endswith("17 files, for camera 1, camera 2 and both.")
    assert (out / f"{name}-cameras-report.pdf").read_bytes().startswith(b"%PDF")
    fs = cv2.FileStorage(str(out / f"{name}-cameras.yml"), cv2.FILE_STORAGE_READ)
    assert np.allclose(fs.getNode("R").mat(), rig.rotation) and np.allclose(fs.getNode("T").mat().ravel(), rig.translation)
    assert np.allclose(fs.getNode("K2").mat(), rig.second.camera_matrix)
    assert np.allclose(fs.getNode("D1").mat().ravel(), rig.first.distortion)
    assert fs.getNode("distortion_model_1").string()
    tables = np.load(out / f"{name}-cameras-rectify-maps.npz")
    assert np.allclose(fs.getNode("P2").mat(), tables["P2"]) and np.allclose(fs.getNode("Q").mat(), tables["Q"])
    assert tables["map1_x"].shape == (rig.first.image_height, rig.first.image_width)
    fs.release()
    both = json.loads((out / f"{name}-cameras.json").read_text())
    assert both["apart_mm"] == pytest.approx(rig.apart_mm) and len(both["pairs"]) == 8
    assert both["camera_files"] == {"camera_1": f"{name}-cam1-camera.json", "camera_2": f"{name}-cam2-camera.json"}
    assert both["camera_2"]["camera_matrix"] == json.loads((out / f"{name}-cam2-camera.json").read_text())["camera_matrix"]
    assert "x2 = rotation @ x1" in both["convention"]
    assert both["rectification"]["aligned_along"] == "rows" and np.allclose(both["rectification"]["Q"], tables["Q"])
    # The values do what the file says: the two cameras' views of the board agree through R and T.
    R, T = np.array(both["rotation"]), np.array(both["translation_mm"])
    v1 = next(v for v in rig.first.views if v.index == 0)
    v2 = next(v for v in rig.second.views if v.index == 0)
    R1 = cv2.Rodrigues(v1.rvec)[0]
    assert np.allclose(R @ np.asarray(v1.tvec).ravel() + T, np.asarray(v2.tvec).ravel(), atol=0.5)
    assert np.allclose(R @ R1, cv2.Rodrigues(v2.rvec)[0], atol=2e-3)
    # Same result exported again is the same revision; a changed one gets -r2 and leaves the first alone.
    assert bench.project.revision == 1
    bench.export_to(str(out), formats={"json"})
    assert bench.project.revision == 1
    bench._toggle(1)
    settle(bench)
    bench.export_to(str(out), formats={"json"})
    assert bench.project.revision == 2 and (out / f"{name}-r2-cam1-camera.json").exists()
    assert (out / f"{name}-r2-cameras.yml").exists() and (out / f"{name}-cameras.yml").exists()
    bench._toggle(1)
    settle(bench)

    # The project file keeps the second camera.
    again = Project.load(str(out / f"{name}.plumbline"))
    assert again.second_camera and len(again.second_image_paths) == 8 and len(again.second_detections) == 8
    win.open_path(str(out / f"{name}.plumbline"))
    settle(win.bench)
    # The file is the one written with revision 2, when pair 2 was left out.
    assert win.bench.rig is not None and win.bench.rig.unpaired == [1] and win.bench.project.revision == 2
    win.new_project()
    assert win.bench.cams_box.isHidden() and win.bench.table.columnCount() == 3 and win.bench.rig is None

    # All images handed over together: the file names tell the cameras apart.
    bench = win.bench
    bench._fill_fields(charuco)
    bench._target_changed()
    bench.add_images(sorted(one + two[:7], key=lambda p: p[::-1]))
    assert bench.project.second_camera and len(bench.project.image_paths) == len(bench.project.second_image_paths) == 7
    settle(bench)
    assert bench.rig is not None and len(bench.rig.pairs) == 7
    assert bench.status.text().endswith("1 image without a partner was left out.")
    assert bench.rig.apart_mm == pytest.approx(np.linalg.norm(T_TRUE), abs=0.5)

    # The same by dropping the folder onto the window.
    from PySide6.QtCore import QMimeData, QUrl

    class Drop:
        def __init__(self, paths):
            self.data = QMimeData()
            self.data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
            self.accepted = False

        def mimeData(self):
            return self.data

        def acceptProposedAction(self):
            self.accepted = True

    (tmp_path / "notes.txt").write_text("not an image")
    win.new_project()
    bench = win.bench
    bench._fill_fields(charuco)
    bench._target_changed()
    drop = Drop([tmp_path])
    bench.dragEnterEvent(drop)
    bench.dropEvent(drop)
    assert drop.accepted and bench.project.second_camera and len(bench.project.second_image_paths) == 8
    settle(bench)
    assert bench.rig is not None and len(bench.rig.pairs) == 8

    # One camera's images dropped: still one camera.
    win.new_project()
    bench = win.bench
    bench._fill_fields(charuco)
    bench._target_changed()
    bench.dropEvent(Drop(one))
    assert not bench.project.second_camera and len(bench.project.image_paths) == 8
    settle(bench)
    assert bench.rig is None and bench.result is not None
    QSettings().setValue("project_counter", 0)


def test_two_different_lenses_in_the_window(tmp_path, charuco, camera, fisheye_camera):
    """Camera 1 has a standard lens, camera 2 a fisheye: camera 2 gets its own lens settings."""
    pytest.importorskip("PySide6")
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app
    from plumbline.core.project import Project

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    rng = np.random.default_rng(7)
    paths = []
    for i, (r, t) in enumerate(synth.sample_poses(charuco, camera, 10, 7)):
        for cam, pose, name in ((camera, (r, t), "cam1"), (fisheye_camera, second_pose(r, t), "cam2")):
            paths.append(str(tmp_path / f"{name}-{i:02d}.png"))
            cv2.imwrite(paths[-1], synth.render(charuco, cam, pose[0], pose[1], rng))

    def settle(bench):
        end, quiet = time.time() + 240, 0
        while time.time() < end and quiet <= 40:
            app.processEvents()
            time.sleep(0.02)
            quiet = 0 if bench._busy() else quiet + 1

    win = MainWindow()
    win.show()
    win.new_project()
    bench = win.bench
    bench._fill_fields(charuco)
    bench._target_changed()
    assert bench.own_box.isHidden()
    bench.add_images(paths)
    settle(bench)
    # One set of settings for both to begin with.
    assert not bench.own_box.isHidden() and not bench.own_model.isChecked() and bench.model_for.isHidden()
    assert not bench.rig.second.fisheye and not bench.rig.first.fisheye

    bench.own_model.setChecked(True)
    assert bench.project.second_options == bench.project.options and bench.rig is None
    bench._select_camera(1)
    assert "camera 2" in bench.model_for.text() and not bench.model_for.isHidden()
    bench.lens_kind.setCurrentIndex(1)
    assert bench.project.second_options.fisheye and not bench.project.options.fisheye
    bench._select_camera(0)
    assert bench.lens_kind.currentIndex() == 0          # camera 1's settings are shown again, unchanged
    bench.run_calibration()
    settle(bench)
    rig = bench.rig
    assert rig.second.fisheye and not rig.first.fisheye
    assert rig.second.rms_point < 0.2 and rig.rms_point < 0.2
    print("fisheye fx", rig.second.fx, "off", off(rig), "std", rig.apart_std_mm, rig.turned_std_deg)
    # The target fills only the middle of the wide camera's image here, so its focal length is loosely held.
    assert rig.second.fx == pytest.approx(fisheye_camera.K[0, 0], rel=0.01)
    print("std of T", rig.std[3:], "std of rotation, deg", np.rad2deg(rig.std[:3]))
    assert off(rig)[0] < 3 * np.linalg.norm(rig.std[3:]) + 0.3
    assert off(rig)[1] < 3 * np.rad2deg(np.linalg.norm(rig.std[:3])) + 0.05

    # The board setting is one for both cameras, whichever camera is selected.
    bench._select_camera(1)
    bench.opt_board.setChecked(True)
    assert bench.project.options.release_board and bench.project.second_options.release_board
    bench.opt_board.setChecked(False)

    # Saved and opened again, camera 2 keeps its own settings.
    bench.project.save(str(tmp_path / "two-lenses.plumbline"))
    again = Project.load(str(tmp_path / "two-lenses.plumbline"))
    assert again.second_options.fisheye and not again.options.fisheye
    win.open_path(str(tmp_path / "two-lenses.plumbline"))
    settle(win.bench)
    assert win.bench.own_model.isChecked() and win.bench.rig.second.fisheye
    # Taking the tick away puts both cameras on camera 1's settings.
    win.bench.own_model.setChecked(False)
    assert win.bench.project.second_options is None
    QSettings().setValue("project_counter", 0)
