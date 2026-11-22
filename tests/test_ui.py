"""The windows are built and driven without a display."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")


def _wait(app, win, limit=240):
    end = time.time() + limit
    quiet = 0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)
        busy = (win._worker is not None and win._worker.isRunning()) or win.bench._busy()
        quiet = 0 if busy else quiet + 1
        if quiet > 40:
            return
    raise AssertionError("the app did not go idle")


def test_sample_images_run_from_start_window_to_export(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app
    from plumbline.core.project import Project

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    win = MainWindow()
    win.show()
    assert win.stack.currentWidget() is win.welcome

    win.try_sample()
    _wait(app, win)
    bench = win.bench
    assert win.stack.currentWidget() is bench
    r = bench.result
    assert r is not None and len(r.views) >= 18
    assert abs(r.fx - 1500) / 1500 < 0.005
    assert r.rms_point < 0.5
    assert bench.steps[4]._status.text() == "Files and report"

    first = r.views[2].index
    bench._toggle(first)
    _wait(app, win)
    assert first in bench.result.excluded
    assert bench.table.item(first, 2).text() == "out"

    from plumbline.ui.export_dialog import ExportDialog

    # The export window shows the files before they are written and follows the name typed.
    dialog = ExportDialog(bench.project, bench.result, bench)
    dialog.folder_edit.setText(str(tmp_path))
    assert dialog.revision == 1 and dialog.ok.isEnabled()
    assert dialog.file_names()["report"] == "Sample-ChArUco-report.pdf"
    dialog.name_edit.setText("line 3: left/cam")
    assert dialog.project_name == "line-3-leftcam"
    assert dialog.file_names()["opencv_yaml"] == "line-3-leftcam-camera.yml"
    assert "removed" in dialog.files_note.text()
    assert "opencv_xml" not in dialog.formats() and "project" in dialog.formats()
    dialog._checks["opencv_xml"].setChecked(True)
    dialog._checks["json"].setChecked(False)
    assert dialog.file_names()["opencv_xml"] == "line-3-leftcam-camera.xml" and "json" not in dialog.file_names()
    dialog.folder_edit.setText(str(tmp_path / "missing"))
    assert not dialog.ok.isEnabled()
    dialog.reject()

    def accept_as_is(self):
        self.folder_edit.setText(str(tmp_path))
        return ExportDialog.DialogCode.Accepted

    monkeypatch.setattr(ExportDialog, "exec", accept_as_is)
    bench.export_all()
    first_set = ("Sample-ChArUco-camera.yml", "Sample-ChArUco-camera.json", "Sample-ChArUco-camera_info.yaml",
                 "Sample-ChArUco-report.pdf", "Sample-ChArUco.plumbline")
    for name in first_set:
        assert (tmp_path / name).stat().st_size > 100
    pdf = (tmp_path / "Sample-ChArUco-report.pdf").read_bytes()
    assert pdf.startswith(b"%PDF")
    assert "camera_name: Sample-ChArUco" in (tmp_path / "Sample-ChArUco-camera_info.yaml").read_text()
    assert bench.project.revision == 1

    # Only the ticked formats are written; XML is one of the choices.
    (tmp_path / "x").mkdir()
    bench.export_to(str(tmp_path / "x"), formats={"opencv_xml"})
    assert sorted(p.name for p in (tmp_path / "x").iterdir()) == ["Sample-ChArUco-camera.xml", "Sample-ChArUco.plumbline"]
    bench.project.path = str(tmp_path / "Sample-ChArUco.plumbline")

    # The same result again is the same revision: nothing new appears.
    bench.export_all()
    assert bench.project.revision == 1
    assert not list(tmp_path.glob("*-r2-*"))

    # A changed result is revision 2, written beside revision 1, which stays as it was.
    before = (tmp_path / "Sample-ChArUco-camera.yml").read_text()
    second = bench.result.views[5].index
    bench._toggle(second)
    _wait(app, win)
    bench.export_all()
    assert bench.project.revision == 2
    for name in ("Sample-ChArUco-r2-camera.yml", "Sample-ChArUco-r2-report.pdf"):
        assert (tmp_path / name).stat().st_size > 100
    assert (tmp_path / "Sample-ChArUco-camera.yml").read_text() == before
    again = Project.load(str(tmp_path / "Sample-ChArUco.plumbline"))
    assert again.excluded == {first, second} and len(again.detections) == 24 and again.revision == 2

    # A reminder set at export shows up on the start window once its date has come.
    from plumbline.ui import welcome as welcome_module

    assert bench.project.recheck_months == 6 and bench.project.exported_on  # the export window starts at 6 months
    bench.export_to(str(tmp_path), recheck_months=6)
    assert bench.project.due_date() > bench.project.exported_on
    items = welcome_module.recent_projects()
    assert items[0]["due"] == bench.project.due_date()
    assert not welcome_module.is_due(items[0]["due"])
    welcome_module.remember_project(bench.project.path, "ChArUco", 0.05, "2026-01-01")
    win.show_welcome()
    texts = [w.text() for w in win.welcome.findChildren(type(win.welcome.findChildren(welcome_module.QLabel)[0]))]
    assert any("due for a new calibration" in t for t in texts)
    assert any(t.startswith("Recalibrate, due 2026-01-01") for t in texts)
    assert welcome_module.question_mailto().startswith("mailto:hello@calib.tools?subject=Plumbline%20question")
    win.stack.setCurrentWidget(bench)

    # Picking a part number fills the pattern in; editing a field afterwards drops the part number.
    win.new_project()
    fresh = win.bench
    fresh.preset.setCurrentIndex(fresh.preset.findData("CT-CHA-P-400X300-D5"))
    assert fresh.project.target.part_number == "CT-CHA-P-400X300-D5"
    assert (fresh.sx.value(), fresh.sy.value(), fresh.square.value(), fresh.marker.value()) == (12, 9, 30.0, 22.0)
    assert fresh.dictionary.currentText() == "DICT_5X5_100"
    assert "CT-CHA-P-400X300-D5" in fresh.preset_link.text()
    assert fresh.steps[0]._status.text() == "CT-CHA-P-400X300-D5"
    fresh.square.setValue(31.0)
    assert fresh.project.target.part_number is None and fresh.preset.currentIndex() == 0
    QSettings().setValue("project_counter", 0)

    # New projects are numbered in order, and a number is not handed out twice.
    win.new_project()
    assert win.bench.project.name == "Calibration-001"
    win.new_project()
    assert win.bench.project.name == "Calibration-002"

    win.show_welcome()
    assert win.stack.currentWidget() is win.welcome
    win.open_path(str(tmp_path / "Sample-ChArUco.plumbline"))
    _wait(app, win)
    assert win.bench.result is not None and first in win.bench.result.excluded
    assert win.bench.project.name == "Sample-ChArUco" and win.bench.project.revision == 2
    win.close()


def test_fisheye_lens_from_images_to_export(tmp_path, charuco, fisheye_set):
    import cv2
    import yaml
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app
    from plumbline.core.project import Project

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    _poses, images, _dets = fisheye_set
    shots = tmp_path / "shots"
    shots.mkdir()
    paths = []
    for i, img in enumerate(images):
        paths.append(str(shots / f"wide-{i:02d}.png"))
        cv2.imwrite(paths[-1], img)

    win = MainWindow()
    win.show()
    win.new_project()
    bench = win.bench
    bench._fill_fields(charuco)
    bench._target_changed()
    assert not bench.view.corrected_btn.isVisible()  # nothing to correct with yet

    # Standard is the starting choice; its extra terms disappear when the fisheye model is picked.
    assert bench.lens_kind.currentData() is False and not bench.opt_k3.isHidden()
    bench.lens_kind.setCurrentIndex(1)
    assert bench.project.options.fisheye and bench.opt_k3.isHidden()
    assert "fisheye" in bench.model_text.text()

    bench.add_images(paths)
    _wait(app, win)
    r = bench.result
    assert r is not None and r.fisheye and len(r.distortion) == 4
    assert abs(r.fx - 560) / 560 < 0.005 and r.rms_point < 0.3

    # The corrected view is offered once there is a result, and changes what is drawn.
    bench.table.selectRow(1)
    app.processEvents()
    assert bench.view.corrected_btn.isVisible() and not bench.view.corrected
    plain = bench.view.grab().toImage()
    bench.view.corrected_btn.setChecked(True)
    app.processEvents()
    assert bench.view.corrected and "corrected" in bench.view._caption
    assert bench.view.grab().toImage() != plain

    out = tmp_path / "out"
    out.mkdir()
    bench.export_to(str(out), formats={"report", "json", "ros", "xlsx", "maps", "example"})
    stem = bench.project.name
    # The example needs the YAML it reads, so that file is written with it.
    assert (out / f"{stem}-camera.yml").exists() and (out / f"{stem}-undistort-maps.npz").stat().st_size > 10000
    assert f"{stem}-camera.yml" in (out / f"{stem}-example.py").read_text()
    from plumbline.ui.export_dialog import ExportDialog
    dialog = ExportDialog(bench.project, bench.result, bench)
    dialog._checks["opencv_yaml"].setChecked(False)
    dialog._checks["example"].setChecked(True)
    assert dialog._checks["opencv_yaml"].isChecked() and "example" in dialog.file_names()
    dialog.reject()
    ros = yaml.safe_load((out / f"{stem}-camera_info.yaml").read_text())
    assert ros["distortion_model"] == "equidistant"
    assert (out / f"{stem}-report.pdf").read_bytes().startswith(b"%PDF")
    assert (out / f"{stem}-data.xlsx").stat().st_size > 1000

    # The model travels with the project file.
    again = Project.load(str(out / f"{stem}.plumbline"))
    assert again.options.fisheye
    win.open_path(str(out / f"{stem}.plumbline"))
    _wait(app, win)
    assert win.bench.lens_kind.currentData() is True and win.bench.result.fisheye

    # Going back to the standard model drops the result and takes the switch away.
    win.bench.lens_kind.setCurrentIndex(0)
    app.processEvents()
    assert win.bench.result is None and not win.bench.view.corrected
    QSettings().setValue("project_counter", 0)
