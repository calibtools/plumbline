"""Export step: name the project, pick the folder, see the files before they are written."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLineEdit, QVBoxLayout,
)

from ..core import export
from ..core.calibrate import CalibrationResult
from ..core.naming import export_files, group_signature, result_signature, rig_files, rig_signature, safe_name
from ..core.project import Project
from . import theme as T
from .widgets import button, frame, label, rule

RECHECK_CHOICES = (("No reminder", 0), ("In 3 months", 3), ("In 6 months", 6), ("In 12 months", 12), ("In 24 months", 24))
DEFAULT_FORMATS = {"report", "opencv_yaml", "json", "ros"}  # the others are offered, off until ticked once

KINDS = (
    ("report", "Report, PDF"),
    ("opencv_yaml", "OpenCV YAML"),
    ("opencv_xml", "OpenCV XML, with per-image figures"),
    ("json", "JSON, with per-image detail"),
    ("ros", "ROS camera_info"),
    ("xlsx", "Excel, every image and corner"),
    ("maps", "Lookup tables that remove the distortion"),
    ("example", "Python example that reads the YAML"),
    ("project", "Plumbline project"),
)


class ExportDialog(QDialog):
    def __init__(self, project: Project, result: CalibrationResult, parent=None, rig=None, group=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setModal(True)
        self.setFixedWidth(640)
        self.setStyleSheet(f"QDialog {{ background: {T.PAPER}; }}")
        self._project, self._result, self._rig, self._group = project, result, rig, group
        self._several = rig is not None or group is not None
        self._revision = project.revision_for(rig_signature(rig) if rig is not None else group_signature(group)
                                              if group is not None else result_signature(result))
        cameras = (rig.first, rig.second) if rig is not None else group.cameras if group is not None else (result,)
        self._ros = all(export.ros_distortion_model(c) is not None for c in cameras)

        v = QVBoxLayout(self)
        v.setContentsMargins(28, 20, 28, 20)
        v.setSpacing(0)
        v.addWidget(label("EXPORT", "eyebrow"))
        v.addSpacing(6)
        v.addWidget(label(f"Revision {self._revision}", font=T.mono(20, 600)))
        v.addSpacing(4)
        self.revision_note = label("", "muted", wrap=True)
        v.addWidget(self.revision_note)
        v.addSpacing(14)

        v.addWidget(label("Project name"))
        v.addSpacing(6)
        self.name_edit = QLineEdit(project.name)
        self.name_edit.setFixedHeight(36)
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
        self.folder_edit.setFixedHeight(36)
        browse = button("Browse")
        browse.setStyleSheet("min-height: 0;")
        browse.setFixedHeight(36)
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(browse)
        v.addLayout(row)
        v.addSpacing(14)

        remind = QHBoxLayout()
        remind.setSpacing(12)
        remind.addWidget(label("Remind me to calibrate this camera again"), 1)
        self.recheck = QComboBox()
        self.recheck.setFixedSize(200, 36)
        for text, months in RECHECK_CHOICES:
            self.recheck.addItem(text, months)
        saved = project.recheck_months if project.revision else int(QSettings().value("recheck_months", 6) or 0)
        self.recheck.setCurrentIndex(max(0, self.recheck.findData(saved)))
        remind.addWidget(self.recheck)
        v.addLayout(remind)
        v.addSpacing(4)
        v.addWidget(label("The reminder appears on the start window of this computer. Nothing is sent anywhere.", "small"))
        v.addSpacing(14)

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
            tick.setFixedSize(20, 24)
            tick.setStyleSheet("min-height: 0;")
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
        v.addSpacing(14)
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
        """The files to be written. With two cameras: camera 1's names, and the two files about both."""
        files = export_files(self.project_name, self._revision, 1 if self._several else None)
        chosen = {k: v for k, v in files.items() if k in self.formats()}
        if self._rig is not None:
            chosen.update(rig_files(self.project_name, self._revision))
        elif self._group is not None:      # three or more cameras: the two files about all of them
            both = rig_files(self.project_name, self._revision)
            chosen.update({k: both[k] for k in ("rig_json", "rig_yaml", "rig_report")})
        return chosen

    @property
    def recheck_months(self) -> int:
        return int(self.recheck.currentData() or 0)

    def accept(self):
        settings = QSettings()
        settings.setValue("recheck_months", self.recheck_months)
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
        if self._checks["example"].isChecked() and not self._checks["opencv_yaml"].isChecked():
            self._checks["opencv_yaml"].setChecked(True)  # the example reads this file
            return
        files = export_files(self.project_name, self._revision, 1 if self._several else None)
        files["project"] = export_files(self.project_name, self._revision)["project"]
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
        if self._rig is not None:
            both = rig_files(self.project_name, self._revision)
            self.revision_note.setText(
                self.revision_note.text() + " Two cameras: the files ticked below are written for cam1 and again "
                f"for cam2. {both['rig_json']} and {both['rig_yaml']} say how camera 2 sits relative to camera 1; "
                "with the report ticked, one more report covers both cameras; with the lookup tables or the "
                "example ticked, one more file lines the two images up row by row.")
        if self._group is not None:
            both = rig_files(self.project_name, self._revision)
            k = len(self._group.cameras)
            self.revision_note.setText(
                self.revision_note.text() + f" {k} cameras: the files ticked below are written once for each camera, "
                f"cam1 to cam{k}. {both['rig_json']} and {both['rig_yaml']} say where each camera sits relative to "
                "camera 1; with the report ticked, one more report covers all cameras. Row-aligned images are written "
                "for two cameras only.")
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
