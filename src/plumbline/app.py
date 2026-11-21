"""Application entry point."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, QProgressDialog, QStackedWidget,
)

from . import __version__
from .core import synth
from .core.naming import default_name
from .core.project import SUFFIX, Project
from .core.targets import CHARUCO, Target
from .ui import theme as T
from .ui.welcome import WelcomePage, remember_project
from .ui.widgets import Worker
from .ui.workbench import Workbench

SAMPLE_TARGET = Target(CHARUCO, 11, 8, 30.0, 22.0, "DICT_5X5_250")
SAMPLE_COUNT = 24


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plumbline by Calib.Tools")
        self.setWindowIcon(QIcon(T.asset("app-icon.png")))
        self.resize(1440, 900)
        self.setMinimumSize(1360, 760)  # the header needs this much for steps, project name and link
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.welcome = WelcomePage()
        self.bench = Workbench()
        self.stack.addWidget(self.welcome)
        self.stack.addWidget(self.bench)
        self.welcome.new_project.connect(self.new_project)
        self.welcome.open_project.connect(self.open_project)
        self.welcome.open_recent.connect(self.open_path)
        self.welcome.try_sample.connect(self.try_sample)
        self.bench.go_home.connect(self.show_welcome)
        self.bench.project_saved.connect(self._saved)
        self._worker = None

    def show_welcome(self):
        self.welcome.refresh_recent()
        self.stack.setCurrentWidget(self.welcome)

    def new_project(self):
        from PySide6.QtCore import QSettings

        settings = QSettings()
        number = int(settings.value("project_counter", 0) or 0) + 1
        settings.setValue("project_counter", number)  # a number is used once, even if the project is dropped
        self.bench.reset()
        self.bench.project.title = default_name(number)
        self.bench._refresh()
        self.stack.setCurrentWidget(self.bench)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", f"Plumbline project (*{SUFFIX})")
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        try:
            project = Project.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Plumbline", f"Could not open this project.\n\n{exc}")
            return
        missing = [p for p in project.image_paths if not Path(p).exists()]
        self.bench.load_project(project)
        self.stack.setCurrentWidget(self.bench)
        if missing:
            QMessageBox.information(
                self, "Plumbline",
                f"{len(missing)} of the project's images are no longer where they were. "
                "The stored detections are still used; those images cannot be shown.",
            )

    def try_sample(self):
        folder = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)) / "sample-charuco"
        paths = [folder / f"sample-{i + 1:02d}.png" for i in range(SAMPLE_COUNT)]
        if all(p.exists() for p in paths):
            self._open_sample(paths)
            return
        folder.mkdir(parents=True, exist_ok=True)
        dialog = QProgressDialog("Rendering the sample images …", "", 0, SAMPLE_COUNT, self)
        dialog.setCancelButton(None)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)

        def job(progress):
            import numpy as np

            camera = synth.default_camera()
            rng = np.random.default_rng(1007)
            for i, (r, t) in enumerate(synth.sample_poses(SAMPLE_TARGET, camera, SAMPLE_COUNT)):
                img = synth.render(SAMPLE_TARGET, camera, r, t, rng)
                ok, buf = cv2.imencode(".png", img)
                buf.tofile(str(paths[i]))
                progress(i + 1, SAMPLE_COUNT)
            return paths

        w = Worker(job, self)
        w.progress.connect(lambda a, _b: dialog.setValue(a))
        w.finished_ok.connect(lambda p: (dialog.close(), self._open_sample(p)))
        w.failed.connect(lambda m: (dialog.close(), QMessageBox.warning(self, "Plumbline", m)))
        self._worker = w
        w.start()

    def _open_sample(self, paths):
        self.bench.load_project(Project(target=SAMPLE_TARGET, title="Sample-ChArUco"))
        self.stack.setCurrentWidget(self.bench)
        self.bench.add_images([str(p) for p in paths])

    def _saved(self, path: str, target: str, rms, due=None):
        remember_project(path, target, rms, due)


def create_app(argv=None) -> QApplication:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setOrganizationName("Calib.Tools")
    app.setOrganizationDomain("calib.tools")
    app.setApplicationName("Plumbline")
    app.setApplicationVersion(__version__)
    T.load_fonts()
    app.setFont(T.sans(14))
    app.setStyleSheet(T.STYLE)
    return app


def main() -> int:
    app = create_app()
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
