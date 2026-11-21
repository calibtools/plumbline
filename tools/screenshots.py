"""Render the windows off screen and save PNGs, to check layout without a display.

    QT_QPA_PLATFORM=offscreen python tools/screenshots.py out_dir
"""
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from plumbline.app import MainWindow, create_app


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def wait_idle(app, win, limit=180):
    end = time.time() + limit
    pump(app, 0.5)
    while time.time() < end:
        busy = (win._worker is not None and win._worker.isRunning()) or win.bench._busy()
        if not busy:
            pump(app, 0.5)
            if not ((win._worker is not None and win._worker.isRunning()) or win.bench._busy()):
                return
        pump(app, 0.2)
    raise SystemExit("timed out waiting for the app to go idle")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots")
    out.mkdir(parents=True, exist_ok=True)
    app = create_app()
    QSettings().clear()
    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    pump(app, 0.5)
    win.grab().save(str(out / "00-welcome.png"))
    win.new_project()
    pump(app, 0.3)
    win.grab().save(str(out / "01-empty.png"))
    win.show_welcome()
    win.try_sample()
    wait_idle(app, win)
    win.grab().save(str(out / "03-result.png"))
    bench = win.bench
    bench.tabs.setCurrentIndex(0)
    pump(app, 0.2)
    win.grab().save(str(out / "02-target-tab.png"))
    bench.tabs.setCurrentIndex(1)
    pump(app, 0.2)
    win.grab().save(str(out / "02-model-tab.png"))
    r = bench.result
    print("RESULT views", len(r.views), "rms", round(r.rms_point, 4), "fx", round(r.fx, 2), "fy", round(r.fy, 2))
    bench._toggle(3)
    wait_idle(app, win)
    win.grab().save(str(out / "04-excluded.png"))
    print("AFTER EXCLUDE views", len(bench.result.views), "rms", round(bench.result.rms_point, 4))
    bench.table.selectRow(1)
    bench.view.corrected_btn.setChecked(True)
    pump(app, 0.3)
    win.grab().save(str(out / "05-corrected.png"))
    bench.view.corrected_btn.setChecked(False)

    # A wide lens, fitted with the fisheye model.
    import cv2
    import numpy as np

    from plumbline.app import SAMPLE_TARGET
    from plumbline.core import synth

    camera = synth.fisheye_camera()
    rng = np.random.default_rng(1007)
    folder = out / "wide-lens-images"
    folder.mkdir(exist_ok=True)
    paths = []
    for i, (r, t) in enumerate(synth.sample_poses(SAMPLE_TARGET, camera, 24)):
        paths.append(str(folder / f"wide-{i + 1:02d}.png"))
        cv2.imwrite(paths[-1], synth.render(SAMPLE_TARGET, camera, r, t, rng))
    win.new_project()
    bench._fill_fields(SAMPLE_TARGET)
    bench._target_changed()
    bench.lens_kind.setCurrentIndex(1)
    bench.tabs.setCurrentIndex(1)
    pump(app, 0.2)
    win.grab().save(str(out / "06-fisheye-model-tab.png"))
    bench.add_images(paths)
    wait_idle(app, win)
    bench.tabs.setCurrentIndex(2)
    bench.table.selectRow(1)
    pump(app, 0.3)
    win.grab().save(str(out / "07-fisheye-result.png"))
    r = bench.result
    print("FISHEYE views", len(r.views), "rms", round(r.rms_point, 4), "fx", round(r.fx, 2), "dist", np.round(r.distortion, 5))
    bench.view.corrected_btn.setChecked(True)
    pump(app, 0.3)
    win.grab().save(str(out / "08-fisheye-corrected.png"))
    bench.export_to(str(out), formats={"report", "opencv_yaml", "ros"})

    # A stock asymmetric circle grid, picked by part number.
    from plumbline.core.catalog import find_board

    board = find_board("CT-CIR-P-300X200-ASY").target
    camera = synth.default_camera()
    rng = np.random.default_rng(1007)
    folder = out / "circle-grid-images"
    folder.mkdir(exist_ok=True)
    paths = []
    for i, (r, t) in enumerate(synth.sample_poses(board, camera, 24)):
        paths.append(str(folder / f"dots-{i + 1:02d}.png"))
        cv2.imwrite(paths[-1], synth.render(board, camera, r, t, rng))
    win.new_project()
    bench.preset.setCurrentIndex(bench.preset.findData("CT-CIR-P-300X200-ASY"))
    bench.tabs.setCurrentIndex(0)
    pump(app, 0.2)
    win.grab().save(str(out / "10-circles-target-tab.png"))
    bench.add_images(paths)
    wait_idle(app, win)
    bench.tabs.setCurrentIndex(2)
    bench.table.selectRow(next(v.index for v in bench.result.views))
    pump(app, 0.3)
    win.grab().save(str(out / "11-circles-result.png"))
    r = bench.result
    print("CIRCLES views", len(r.views), "rms", round(r.rms_point, 4), "fx", round(r.fx, 2))
    assert not bench.view.corrected, "a new project must not open in the corrected view"
    bench.table.selectRow(next(i for i, d in enumerate(bench.project.detections) if not d.usable))
    pump(app, 0.3)
    win.grab().save(str(out / "12-circles-partly-outside.png"))
    bench.export_to(str(out), formats={"report"})


if __name__ == "__main__":
    main()
