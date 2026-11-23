"""Checking a two-camera result with pairs that took no part in it: the board as a ruler."""
import numpy as np
import pytest

from plumbline.core import synth
from plumbline.core.calibrate import ModelOptions
from plumbline.core.check import check_pairs, locate
from plumbline.core.detect import Detection, detect
from plumbline.core.rig import calibrate_pair
from test_rig import T_TRUE, exact_pairs, rendered_pairs, second_pose


def test_lengths_on_pairs_that_took_no_part(charuco, camera):
    first, second = rendered_pairs(charuco, camera, 16, 7)
    rig = calibrate_pair(first, second, charuco)
    # New poses, new noise: seed 21 gives a different set of views than the calibration used.
    held1, held2 = rendered_pairs(charuco, camera, 8, 21)
    result = check_pairs(rig, held1, held2, charuco)
    print("scale", result.scale, "length rms", result.length_error_rms_mm, "max", result.length_error_max_mm,
          "flatness", result.flatness_rms_mm, "spans", result.spans, "longest", result.longest_mm)
    assert len(result.pairs) + len(result.skipped) == 8 and len(result.pairs) >= 6
    assert result.scale == pytest.approx(1.0, abs=0.002)
    assert result.length_error_rms_mm < 0.15 and result.flatness_rms_mm < 0.15
    assert result.longest_mm > 250 and result.spans > 1000
    assert all(400 < p.distance_mm < 2500 and p.ray_gap_rms_mm < 0.2 for p in result.pairs)
    assert result.to_dict()["spans_compared"] == result.spans


def test_a_wrong_distance_between_the_cameras_shows_as_a_wrong_scale(charuco, camera):
    first, second = exact_pairs(charuco, camera, False, 12, 3)
    rig = calibrate_pair(first, second, charuco)
    held1, held2 = exact_pairs(charuco, camera, False, 12, 4)
    good = check_pairs(rig, held1[6:], held2[6:], charuco)
    assert good.scale == pytest.approx(1.0, abs=0.002)
    # The same result with the cameras taken to be 1 % further apart: every length comes out 1 % long.
    rig.translation = rig.translation * 1.01
    stretched = check_pairs(rig, held1[6:], held2[6:], charuco)
    assert stretched.scale == pytest.approx(1.01, abs=0.002)
    assert stretched.length_error_rms_mm > 5 * good.length_error_rms_mm


def test_points_are_placed_where_they_are(charuco, camera):
    first, second = exact_pairs(charuco, camera, False, 12, 5, noise=0.0)
    rig = calibrate_pair(first, second, charuco)
    r, t = synth.sample_poses(charuco, camera, 12, 7)[2]
    import cv2
    board = charuco.object_points().astype(np.float64)
    truth = (cv2.Rodrigues(np.asarray(r, np.float64))[0] @ board.T).T + np.asarray(t, np.float64).ravel()
    common = np.intersect1d(first[2].ids, second[2].ids)
    ia = {int(k): j for j, k in enumerate(first[2].ids)}
    ib = {int(k): j for j, k in enumerate(second[2].ids)}
    pts, gap = locate(rig, first[2].corners[[ia[int(k)] for k in common]], second[2].corners[[ib[int(k)] for k in common]])
    assert np.abs(pts - truth[common]).max() < 0.05 and gap.max() < 0.01
    with pytest.raises(ValueError):
        check_pairs(rig, [Detection("a", 10, 10)], [Detection("b", 10, 10)], charuco)


def test_the_check_in_the_window(tmp_path, charuco, camera):
    pytest.importorskip("PySide6")
    import os
    import time

    import cv2

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    rng = np.random.default_rng(7)
    files = []
    for i, (r, t) in enumerate(synth.sample_poses(charuco, camera, 10, 7)):
        for name, pose in (("cam1", (r, t)), ("cam2", second_pose(r, t))):
            files.append(str(tmp_path / f"{name}-{i:02d}.png"))
            cv2.imwrite(files[-1], synth.render(charuco, camera, pose[0], pose[1], rng))
    used, held = files[:14], files[14:]

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
    assert bench.check_part.isHidden()
    bench.add_images(used)
    settle(bench)
    assert bench.rig is not None and not bench.check_part.isHidden() and bench.check_box.isHidden()

    # Images of the calibration itself are refused; images named another way are not guessed at.
    bench.check_with(used[:4])
    assert bench.check is None and "took no part" in bench.status.text()
    bench.check_with([str(tmp_path / f"{side}_{i}.png") for side in ("left", "right") for i in range(3)])
    assert bench.check is None and "cam1 / cam2" not in bench.status.text() and "1 / 2" in bench.status.text()

    bench.check_with(held + used[:2])
    settle(bench)
    c = bench.check
    assert c is not None and len(c.pairs) + len(c.skipped) == 3 and abs(c.scale - 1) < 0.003
    assert not bench.check_box.isHidden() and bench.check_values["Pairs"].text().startswith(f"{len(c.pairs)} used")
    assert bench.status.text().endswith("2 images of the calibration left out.")
    rig = bench.rig

    # The check goes into the file and the report about both cameras.
    import json

    from plumbline.ui.report import write_rig_report

    out = tmp_path / "out"
    out.mkdir()
    bench.export_to(str(out), formats={"json"})
    said = json.loads((out / f"{bench.project.name}-cameras.json").read_text())["check_with_other_images"]
    assert said["measured_over_true"] == c.scale and len(said["pairs"]) == len(c.pairs)
    two = [bench.project.detections, bench.project.second_detections]
    assert write_rig_report(rig, *two, out / "with.pdf", check=c) == write_rig_report(rig, *two, out / "without.pdf")
    assert (out / "with.pdf").stat().st_size > (out / "without.pdf").stat().st_size

    # The check images are kept with the project: a new calibration is checked with them again.
    bench._toggle(0)
    settle(bench)
    assert bench.rig is not rig and bench.check is not None and bench.check is not c
    assert not bench.check_box.isHidden() and len(bench.check.pairs) == len(c.pairs)
    bench.export_to(str(out), formats={"json"})
    name = bench.project.name
    assert "check_with_other_images" in (out / f"{name}-r2-cameras.json").read_text()
    win.open_path(str(out / f"{name}.plumbline"))
    bench = win.bench
    settle(bench)
    assert bench.rig is not None and bench.check is not None and len(bench.project.check_detections[0]) == 3

    # Other target settings, or another number of cameras: the kept images no longer apply.
    bench._add_camera()
    assert bench.project.check_detections == [] and bench.check is None
    QSettings().setValue("project_counter", 0)
