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
# Two of the sample images are spoiled on purpose: the board moved by this much (mm) while the image was read out.
# They look like the others; their error does not. Leaving them out is what the sample is there to show.
SAMPLE_SPOILED = {6: 2.0, 17: -2.0}
SAMPLE_PAIRS = 16
SAMPLE_PAIR_SHIFT = (-120.0, 2.0, 5.0)      # x2 = R x1 + T, mm: camera 2 sits 120 mm to the right of camera 1
SAMPLE_PAIR_TURN = (0.0, -0.0698132, 0.0)   # rotation vector, rad: turned 4 degrees towards camera 1
SAMPLE_HINT = " Two are spoiled on purpose: exclude the two tall bars."


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
        self.welcome.try_sample_pair.connect(self.try_sample_pair)
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
        def images(camera, rng):
            for i, (r, t) in enumerate(synth.sample_poses(SAMPLE_TARGET, camera, SAMPLE_COUNT)):
                if i in SAMPLE_SPOILED:
                    yield synth.render_moving(SAMPLE_TARGET, camera, r, t, SAMPLE_SPOILED[i], rng)
                else:
                    yield synth.render(SAMPLE_TARGET, camera, r, t, rng)

        names = [f"sample-{i + 1:02d}.png" for i in range(SAMPLE_COUNT)]
        self._sample("sample-charuco-2", names, images, lambda paths: self._open_sample(paths, "Sample-ChArUco", SAMPLE_HINT))

    def try_sample_pair(self):
        """Two cameras side by side looking at the same board: camera 2 is 120 mm to the right and turned 4 degrees in."""
        def images(camera, rng):
            import numpy as np

            turn = cv2.Rodrigues(np.array(SAMPLE_PAIR_TURN))[0]
            for r, t in synth.sample_poses(SAMPLE_TARGET, camera, SAMPLE_PAIRS):
                first = cv2.Rodrigues(np.asarray(r, np.float64))[0]
                yield synth.render(SAMPLE_TARGET, camera, r, t, rng)
                yield synth.render(SAMPLE_TARGET, camera, cv2.Rodrigues(turn @ first)[0].ravel(),
                                   turn @ np.asarray(t, np.float64).ravel() + np.array(SAMPLE_PAIR_SHIFT), rng)

        names = [f"cam{c}-{i + 1:02d}.png" for i in range(SAMPLE_PAIRS) for c in (1, 2)]
        self._sample("sample-two-cameras", names, images, lambda paths: self._open_sample(paths, "Sample-two-cameras"))

    def _sample(self, folder_name: str, names: list[str], images, then) -> None:
        """Render a set of sample images once, keep them in the cache folder, and open them."""
        folder = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)) / folder_name
        paths = [folder / name for name in names]
        if all(p.exists() for p in paths):
            then(paths)
            return
        folder.mkdir(parents=True, exist_ok=True)
        dialog = QProgressDialog("Rendering the sample images …", "", 0, len(paths), self)
        dialog.setCancelButton(None)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)

        def job(progress):
            import numpy as np

            for i, img in enumerate(images(synth.default_camera(), np.random.default_rng(1007))):
                ok, buf = cv2.imencode(".png", img)
                buf.tofile(str(paths[i]))
                progress(i + 1, len(paths))
            return paths

        w = Worker(job, self)
        w.progress.connect(lambda a, _b: dialog.setValue(a))
        w.finished_ok.connect(lambda p: (dialog.close(), then(p)))
        w.failed.connect(lambda m: (dialog.close(), QMessageBox.warning(self, "Plumbline", m)))
        self._worker = w
        w.start()

    def _open_sample(self, paths, title: str, hint: str = ""):
        self.bench.load_project(Project(target=SAMPLE_TARGET, title=title))
        self.stack.setCurrentWidget(self.bench)
        self.bench.add_images([str(p) for p in paths])
        if hint:
            self.bench._left_out = hint       # said once, after the first calibration

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
