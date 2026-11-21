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
    from PySide6.QtWidgets import QFileDialog

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

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    bench.export_all()
    for name in ("camera.yml", "camera.json", "camera_info.yaml", "calibration.plumbline", "report.pdf"):
        assert (tmp_path / name).stat().st_size > 100
    pdf = (tmp_path / "report.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") + pdf.count(b"/Type/Page") >= 2
    again = Project.load(str(tmp_path / "calibration.plumbline"))
    assert again.excluded == {first} and len(again.detections) == 24

    win.show_welcome()
    assert win.stack.currentWidget() is win.welcome
    win.open_path(str(tmp_path / "calibration.plumbline"))
    _wait(app, win)
    assert win.bench.result is not None and first in win.bench.result.excluded
    win.close()
