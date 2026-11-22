"""A board that is not flat or not to size: noticed, and estimated together with the camera."""
import json
from dataclasses import replace

import numpy as np
import pytest

from plumbline.core import export, lens, synth
from plumbline.core.board import keep_place_and_size
from plumbline.core.calibrate import ModelOptions, calibrate
from plumbline.core.detect import Detection
from plumbline.core.findings import CHECK, NOTE, findings


def shaped(target, sag_mm=0.0, stretch_x=0.0):
    """The board bent into a shallow bowl and/or stretched along x."""
    p = target.object_points().astype(np.float64)
    c = p[:, :2].mean(axis=0)
    half = np.abs(p[:, :2] - c).max()
    r2 = ((p[:, :2] - c) ** 2).sum(axis=1) / half**2
    p[:, 2] = sag_mm * (r2 - r2.mean())
    p[:, 0] = c[0] + (p[:, 0] - c[0]) * (1 + stretch_x)
    return p


def views(target, board, camera, wide=False, count=30, noise=0.08, seed=5):
    """Exact projections of the true board plus pixel noise; points outside the image are dropped."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (r, t) in enumerate(synth.sample_poses(target, camera, count, seed)):
        p = lens.project(board, r, t, camera.K, camera.dist, wide)
        inside = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
        out.append(Detection(f"v{i}.png", camera.width, camera.height, np.nonzero(inside)[0].astype(np.int32),
                             (p[inside] + rng.normal(0, noise, (int(inside.sum()), 2))).astype(np.float32)))
    return out


def shape_error(result, truth):
    b = result.board
    est, true = b.points[b.estimated], truth[b.estimated]
    return np.abs(keep_place_and_size(est, true) - true).max()


@pytest.mark.parametrize("wide", [False, True])
def test_a_sagging_board_is_noticed_and_estimated(wide, charuco, camera, fisheye_camera):
    cam = fisheye_camera if wide else camera
    f = cam.K[0, 0]
    truth = shaped(charuco, sag_mm=1.0)
    dets = views(charuco, truth, cam, wide)
    assert any(len(d.ids) < charuco.corner_count for d in dets)  # some images show only part of the board
    options = ModelOptions(fisheye=wide)

    # Taken as flat, the focal length is far off although the error is still below half a pixel.
    flat = calibrate(dets, charuco, options)
    assert abs(flat.fx - f) > 4 and flat.rms_point < 0.5
    assert flat.board.clearly_better
    text = next(t for tag, t in findings(flat, 0.9) if tag == CHECK and "out of flat" in t)
    assert "Estimate the shape of the board" in text and "1.4 mm" in text or "1.5 mm" in text

    # With the shape estimated, the camera is right and the shape is found.
    free = calibrate(dets, charuco, replace(options, release_board=True))
    assert abs(free.fx - f) < 0.6 and free.rms_point < 0.12
    assert free.board.settled and free.board.out_of_flat_mm == pytest.approx(truth[:, 2].max() - truth[:, 2].min(), abs=0.1)
    assert shape_error(free, truth) < 0.08
    assert np.allclose(free.board_points(), free.board.points) and not np.allclose(flat.board_points(), free.board.points)
    assert any(tag == NOTE and "was estimated with the camera" in t and "out of flat" in t
               for tag, t in findings(free, 0.9))


def test_a_stretched_board_is_estimated(charuco, camera):
    truth = shaped(charuco, stretch_x=0.003)       # 0.3 % longer along x, as from a printer that does not scale true
    dets = views(charuco, truth, camera)
    flat = calibrate(dets, charuco)
    free = calibrate(dets, charuco, ModelOptions(release_board=True))
    assert flat.board.clearly_better and free.rms_point < 0.8 * flat.rms_point
    assert abs(free.fy / free.fx - 1) < 2e-4 < abs(flat.fy / flat.fx - 1)   # the stretch had gone into fy / fx
    assert shape_error(free, truth) < 0.08 and free.board.in_plane_mm == pytest.approx(0.2, abs=0.08)


def test_a_sound_board_raises_no_alarm(tmp_path, charuco, camera, charuco_set, fisheye_set):
    # Simulated flat board.
    dets = views(charuco, shaped(charuco), camera)
    result = calibrate(dets, charuco)
    assert not result.board.clearly_better and result.board.out_of_flat_mm < 0.2
    assert not any("fit clearly better" in t for _tag, t in findings(result, 0.9))
    # Rendered sample sets, with real detection in them.
    for rendered, options in ((charuco_set[2], ModelOptions()), (fisheye_set[2], ModelOptions(fisheye=True))):
        result = calibrate(rendered, charuco, options)
        assert result.board is not None and not result.board.clearly_better
        assert not any("fit clearly better" in t for _tag, t in findings(result, 0.9))
    # Asked to estimate a board that is fine: it comes out fine, and the findings say it was not needed.
    free = calibrate(charuco_set[2], charuco, ModelOptions(release_board=True))
    assert free.board.out_of_flat_mm < 0.1 and abs(free.fx - 1500) < 1.0
    assert any("was not needed here" in t for _tag, t in findings(free, 0.9))
    export.write_json(free, tmp_path / "camera.json")
    data = json.loads((tmp_path / "camera.json").read_text())
    assert data["options"]["release_board"] is True and len(data["board"]["points_mm"]) == charuco.corner_count
    export.write_json(result, tmp_path / "plain.json")
    plain = json.loads((tmp_path / "plain.json").read_text())
    assert plain["board"] is None and "release_board" not in plain["options"]


def test_a_wrong_lens_model_is_not_blamed_on_the_board(charuco, fisheye_set):
    """A fisheye lens under the standard model also fits better with a bent board; the model is named instead."""
    wrong = calibrate(fisheye_set[2], charuco)
    assert wrong.board.clearly_better and wrong.pattern.has_pattern
    texts = [t for _tag, t in findings(wrong, 0.9)]
    assert not any("fit clearly better" in t for t in texts) and any("not just noise" in t for t in texts)


def test_option_round_trip_and_excel(tmp_path, charuco, camera):
    from openpyxl import load_workbook

    from plumbline.core.export_xlsx import write_xlsx

    assert ModelOptions.from_dict(ModelOptions(release_board=True).to_dict()).release_board
    assert "release_board" not in ModelOptions().to_dict()
    truth = shaped(charuco, sag_mm=1.0)
    dets = views(charuco, truth, camera)
    free = calibrate(dets, charuco, ModelOptions(release_board=True))
    write_xlsx(free, dets, tmp_path / "data.xlsx", "Test", 1)
    rows = list(load_workbook(tmp_path / "data.xlsx")["Board points"].iter_rows(values_only=True))
    assert rows[0][10:13] == ("Estimated dx, mm", "Estimated dy, mm", "Estimated dz, mm")
    dz = np.array([row[12] for row in rows[1:]])
    assert dz.max() - dz.min() == pytest.approx(free.board.out_of_flat_mm, abs=1e-6)


def test_board_option_in_the_window(tmp_path, charuco, charuco_set):
    pytest.importorskip("PySide6")
    import os
    import time

    import cv2
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app
    from plumbline.core.project import Project

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    paths = []
    for i, img in enumerate(charuco_set[1][:12]):
        paths.append(str(tmp_path / f"img-{i:02d}.png"))
        cv2.imwrite(paths[-1], img)

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
    assert not bench.opt_board.isChecked() and not bench.project.options.release_board
    bench.add_images(paths)
    settle(bench)
    assert bench.result is not None and not bench.result.options.release_board

    bench.opt_board.setChecked(True)             # changing the option drops the result, as any model change does
    assert bench.project.options.release_board and bench.result is None
    bench.run_calibration()
    settle(bench)
    r = bench.result
    assert r.options.release_board and r.board is not None and r.board.settled

    bench.export_to(str(tmp_path), formats={"report", "json"})
    again = Project.load(str(tmp_path / f"{bench.project.name}.plumbline"))
    assert again.options.release_board
    win.open_path(str(tmp_path / f"{bench.project.name}.plumbline"))
    settle(win.bench)
    assert win.bench.opt_board.isChecked()
    win.new_project()
    assert not win.bench.opt_board.isChecked()   # a new project starts with the board as entered
    QSettings().setValue("project_counter", 0)
