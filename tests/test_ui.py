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
    assert bench.steps[4]._status.text() == "Parameters and report"

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
