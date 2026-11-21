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


if __name__ == "__main__":
    main()
