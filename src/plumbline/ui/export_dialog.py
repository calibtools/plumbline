"""Export step: name the project, pick the folder, see the files before they are written."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLineEdit, QVBoxLayout,
)

from ..core import export
from ..core.calibrate import CalibrationResult
from ..core.naming import export_files, result_signature, safe_name
from ..core.project import Project
from . import theme as T
from .widgets import button, frame, label, rule

DEFAULT_FORMATS = {"report", "opencv_yaml", "json", "ros"}  # XML and Excel are offered, off until ticked once

KINDS = (
    ("report", "Report, PDF"),
    ("opencv_yaml", "OpenCV YAML"),
    ("opencv_xml", "OpenCV XML, with per-image figures"),
    ("json", "JSON, with per-image detail"),
    ("ros", "ROS camera_info"),
    ("xlsx", "Excel, every image and corner"),
    ("project", "Plumbline project"),
)


class ExportDialog(QDialog):
    def __init__(self, project: Project, result: CalibrationResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setModal(True)
        self.setFixedWidth(640)
        self.setStyleSheet(f"QDialog {{ background: {T.PAPER}; }}")
        self._project, self._result = project, result
        self._revision = project.revision_for(result_signature(result))
        self._ros = export.ros_distortion_model(result) is not None

        v = QVBoxLayout(self)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(0)
        v.addWidget(label("EXPORT", "eyebrow"))
        v.addSpacing(6)
        v.addWidget(label(f"Revision {self._revision}", font=T.mono(20, 600)))
        v.addSpacing(4)
        self.revision_note = label("", "muted", wrap=True)
        v.addWidget(self.revision_note)
        v.addSpacing(20)

        v.addWidget(label("Project name"))
        v.addSpacing(6)
        self.name_edit = QLineEdit(project.name)
        self.name_edit.setFixedHeight(40)
        self.name_edit.setMaxLength(60)
        v.addWidget(self.name_edit)
        v.addSpacing(4)
        v.addWidget(label("Every exported file starts with this name.", "small"))
        v.addSpacing(16)

        v.addWidget(label("Folder"))
        v.addSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.folder_edit = QLineEdit(self._start_folder())
        self.folder_edit.setFixedHeight(40)
        browse = button("Browse")
        browse.setFixedHeight(40)
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(browse)
        v.addLayout(row)
        v.addSpacing(20)

        v.addWidget(label("FILES", "eyebrow"))
        v.addSpacing(8)
        box = frame("sheet")
        self.grid = QGridLayout(box)
        self.grid.setContentsMargins(14, 8, 14, 8)
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(6)
        self._rows = {}
        self._checks = {}
        chosen = QSettings().value("export_formats", None)
        chosen = set(chosen) if isinstance(chosen, (list, tuple)) else (
            {chosen} if isinstance(chosen, str) and chosen else DEFAULT_FORMATS)
        for r, (key, what) in enumerate(KINDS):
            tick = QCheckBox()
            tick.setFixedWidth(20)
            if key == "project":  # always written: it is what reopens the session
                tick.setChecked(True)
                tick.setEnabled(False)
                tick.setToolTip("The project file is always saved.")
            else:
                tick.setChecked(key in chosen and not (key == "ros" and not self._ros))
                tick.setEnabled(not (key == "ros" and not self._ros))
                tick.toggled.connect(self._update)
            name = label("", font=T.mono(12, 600))
            note = label(what, "small")
            self.grid.addWidget(tick, r, 0)
            self.grid.addWidget(name, r, 1)
            self.grid.addWidget(note, r, 2, Qt.AlignmentFlag.AlignRight)
            self._rows[key] = (name, note, what)
            self._checks[key] = tick
        self.grid.setColumnStretch(1, 1)
        v.addWidget(box)
        v.addSpacing(8)
        self.files_note = label("", "small", wrap=True)
        v.addWidget(self.files_note)
        v.addSpacing(20)
        v.addWidget(rule())
        v.addSpacing(16)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        cancel = button("Cancel")
        self.ok = button("Export", "primary")
        cancel.clicked.connect(self.reject)
        self.ok.clicked.connect(self.accept)
        self.ok.setDefault(True)
        buttons.addWidget(cancel)
        buttons.addWidget(self.ok)
        v.addLayout(buttons)

        self.name_edit.textChanged.connect(self._update)
        self.folder_edit.textChanged.connect(self._update)
        self._update()

    # ------------------------------------------------------------------ api
    @property
    def project_name(self) -> str:
        return safe_name(self.name_edit.text())

    @property
    def folder(self) -> str:
        return self.folder_edit.text().strip()

    @property
    def revision(self) -> int:
        return self._revision

    def formats(self) -> set[str]:
        """Which files to write. The project file is always among them."""
        return {k for k, box in self._checks.items() if box.isChecked()}

    def file_names(self) -> dict[str, str]:
        files = export_files(self.project_name, self._revision)
        return {k: v for k, v in files.items() if k in self.formats()}

    def accept(self):
        settings = QSettings()
        settings.setValue("export_folder", self.folder)
        settings.setValue("export_formats", sorted(self.formats() - {"project"}))
        super().accept()

    # -------------------------------------------------------------- private
    def _start_folder(self) -> str:
        if self._project.path:
            return str(Path(self._project.path).parent)
        saved = QSettings().value("export_folder", "")
        if saved and Path(saved).is_dir():
            return saved
        return QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose a folder", self.folder)
        if chosen:
            self.folder_edit.setText(chosen)

    def _update(self, *_):
        files = export_files(self.project_name, self._revision)
        folder = Path(self.folder) if self.folder else None
        folder_ok = bool(folder and folder.is_dir())
        existing = 0
        for key, (name, note, what) in self._rows.items():
            if key == "ros" and not self._ros:
                name.setText("–")
                note.setText("ROS camera_info: not written, ROS has no model for thin prism terms")
                continue
            name.setText(files[key])
            there = folder_ok and (folder / files[key]).exists()
            existing += bool(there and key != "project")
            note.setText(what + (" · replaces the file there" if there else ""))
        typed = self.name_edit.text().strip()
        changed = typed and typed != self.project_name
        same = self._project.revision and self._revision == self._project.revision
        if same:
            self.revision_note.setText("The result has not changed since the last export, so this is the same revision.")
        elif self._revision > 1:
            self.revision_note.setText(
                f"The result changed since revision {self._revision - 1}. The earlier files stay as they are."
            )
        else:
            self.revision_note.setText("First export of this project.")
        notes = []
        if changed:
            notes.append(f"Characters that cannot be in a file name were removed: {self.project_name}.")
        if not folder_ok:
            notes.append("This folder does not exist.")
        elif existing and not same:
            notes.append("Files with these names are already in the folder and will be replaced.")
        self.files_note.setText(" ".join(notes))
        self.files_note.setStyleSheet(f"color: {T.RED_TEXT}; font-size: 12px;" if notes else "")
        self.ok.setEnabled(folder_ok and bool(typed))
