"""Main working window: target, images, detection, calibration, export."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from ..core import export, lens
from ..core.catalog import find_board, same_geometry, stock_boards
from ..core.calibrate import (
    CalibrationError, CalibrationResult, ModelOptions, calibrate, coverage_grid, suggest_exclusions,
)
from ..core.detect import Detection, detect_file, read_gray
from ..core.naming import export_files, result_signature, safe_name
from ..core.export_code import write_example, write_maps
from ..core.points import trusted_suspects
from ..core.export_xlsx import write_xlsx
from ..core.findings import CHECK, NOTE, PASS, error_outside_fit, findings
from ..core.pairing import natural, split_by_camera
from ..core.project import SUFFIX, Project
from ..core.rig import RigResult, calibrate_pair
from ..core.targets import (
    APRILGRID, ARUCO_GRID, ASYMMETRIC_CIRCLES, CHARUCO, CHECKERBOARD, CIRCLES, DICTIONARIES, MARKER_GRIDS, Target,
)
from . import theme as T
from .export_dialog import ExportDialog
from .report import write_report
from .widgets import (
    RESIDUAL_GAIN, CoverageMap, ErrorChart, ImageView, LinkLabel, SvgMark, Worker, button, frame,
    label, open_site, rule,
)

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class StepButton(QPushButton):
    """A numbered step in the header with a one-line status under its name."""

    def __init__(self, title: str):
        super().__init__()
        self._title = title
        self._full = ""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(40)
        self.setMinimumWidth(150)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 3, 10, 3)
        lay.setSpacing(0)
        self._name = label(title)
        self._status = label("")
        for lb in (self._name, self._status):
            lb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            lay.addWidget(lb)
        self.set_state("off", "")

    def set_state(self, kind: str, status: str) -> None:
        bg, line, ink, sub = {
            "done": ("transparent", T.LINE_STRONG, T.INK, T.INK_MUTED),
            "next": (T.RED, T.RED, "#FFFFFF", "#FFFFFF"),
            "off": ("transparent", T.HAIRLINE, T.INK_SOFT, T.INK_SOFT),
        }[kind]
        self.setStyleSheet(
            f"StepButton {{ background: {bg}; border: 1px solid {line}; min-height: 38px; padding: 0; }}"
            f"StepButton:hover {{ border-color: {T.INK if kind != 'next' else T.RED_TEXT}; }}"
        )
        self._name.setStyleSheet(f'font-family: "{T.MONO}"; font-size: 13px; font-weight: 600; color: {ink};')
        self._status.setStyleSheet(f"font-size: 11px; color: {sub};")
        self._full = status
        self._fit()

    def _fit(self) -> None:
        """A text that does not fit is cut with an ellipsis, never mid-letter; the tooltip has all of it."""
        room = max(40, self.width() - 22)
        shown = self._status.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, room)
        self.setToolTip(self._full if shown != self._full else "")
        self._status.setText(shown)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


class Workbench(QWidget):
    go_home = Signal()
    project_saved = Signal(str, str, object, object)  # path, target description, rms, recalibration due date

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.setAcceptDrops(True)
        self.project = Project()
        self.rig: RigResult | None = None       # two cameras: both results and how the cameras sit to each other
        self._result: CalibrationResult | None = None
        self._cam = 0                           # the camera whose images are added to and whose result is shown
        self._left_out = ""                     # said once more after the calibration that follows adding images
        self._worker: Worker | None = None
        self._current = -1

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._header())
        root.addWidget(rule("inkrule"))
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)
        mid.addWidget(self._left())
        mid.addWidget(rule("vrule"))
        mid.addWidget(self._centre(), 1)
        mid.addWidget(rule("vrule"))
        mid.addWidget(self._right())
        root.addLayout(mid, 1)
        root.addWidget(rule("inkrule"))
        root.addWidget(self._footer())
        self._refresh()

    @property
    def result(self) -> CalibrationResult | None:
        """The result on show: the only camera's, or with two cameras the selected one's."""
        return self._result

    @result.setter
    def result(self, value: CalibrationResult | None) -> None:
        self._result = value
        if value is None:
            self.rig = None

    def _result_of(self, camera: int) -> CalibrationResult | None:
        if self.rig is not None:
            return self.rig.second if camera else self.rig.first
        return None if self.project.second_camera else self._result

    # ---------------------------------------------------------------- layout
    def _header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(56)
        h = QHBoxLayout(w)
        h.setContentsMargins(20, 0, 20, 0)
        h.setSpacing(8)
        h.addWidget(SvgMark("wordmark.svg", 28))
        h.addSpacing(8)
        div = rule("vrule")
        div.setFixedHeight(24)
        h.addWidget(div)
        h.addSpacing(8)
        home = QPushButton("PLUMBLINE")
        home.setCursor(Qt.CursorShape.PointingHandCursor)
        home.setToolTip("Back to the start window")
        home.setStyleSheet(f'border: 0; padding: 0; min-height: 0; font-family: "{T.MONO}"; font-size: 14px; font-weight: 600;')
        home.clicked.connect(self.go_home)
        h.addWidget(home)
        h.addSpacing(16)
        self.steps = [StepButton(t) for t in ("1 Target", "2 Images", "3 Detect", "4 Calibrate", "5 Export")]
        # The first step shows the part number and needs room for one such as CT-APR-P-800X600-T36;
        # the others give up a little so that the header still fits a 1360 px window.
        self.steps[0].setFixedWidth(164)
        for s in self.steps[1:]:
            s.setMinimumWidth(126)   # they narrow when a long project name needs the room
            s.setMaximumWidth(140)
        actions = (self._goto_target, self.add_images, self.run_detection, self.run_calibration, self.export_all)
        for s, act in zip(self.steps, actions):
            s.clicked.connect(act)
            h.addWidget(s)
        h.addStretch(1)
        self.name_btn = QPushButton("")
        self.name_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.name_btn.setToolTip("Rename this project. Exported files carry this name.")
        self.name_btn.setStyleSheet(
            f'border: 0; padding: 0 4px; min-height: 0; font-family: "{T.MONO}"; font-size: 13px; '
            f"font-weight: 500; color: {T.INK_MUTED}; text-align: right;")
        self.name_btn.clicked.connect(self.rename_project)
        h.addWidget(self.name_btn)
        h.addSpacing(16)
        h.addWidget(LinkLabel(T.VISIT, T.site_url(where="header"), "monolink"))
        return w

    def _left(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(288)
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 16, 0, 16)
        v.setSpacing(10)
        # Cameras: only shown once there is a second one.
        self.cams_box = QWidget()
        cb = QVBoxLayout(self.cams_box)
        cb.setContentsMargins(16, 0, 16, 2)
        cb.setSpacing(6)
        cb.addWidget(label("CAMERAS", "eyebrow"))
        self.cam_rows = []
        for i in range(2):
            row_ = frame("sheet")
            row_.setCursor(Qt.CursorShape.PointingHandCursor)
            row_.setToolTip("Select this camera: images are added to it, and its own result is shown.")
            row_.mouseReleaseEvent = lambda _e, which=i: self._select_camera(which)
            rh = QHBoxLayout(row_)
            rh.setContentsMargins(10, 7, 10, 7)
            rh.setSpacing(8)
            col = QVBoxLayout()
            col.setSpacing(1)
            col.addWidget(label(f"Camera {i + 1}", font=T.mono(13, 600)))
            sub = label("", "small")
            col.addWidget(sub)
            rh.addLayout(col, 1)
            err = label("", font=T.mono(13, 600))
            rh.addWidget(err, 0, Qt.AlignmentFlag.AlignVCenter)
            cb.addWidget(row_)
            self.cam_rows.append((row_, sub, err))
        v.addWidget(self.cams_box)
        self.cams_rule = rule()
        v.addWidget(self.cams_rule)

        head = QHBoxLayout()
        head.setContentsMargins(16, 0, 16, 0)
        self.images_head = label("IMAGES", "eyebrow")
        head.addWidget(self.images_head)
        head.addStretch(1)
        self.images_count = label("", "small")
        head.addWidget(self.images_count)
        v.addLayout(head)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Corners", "Error, px"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 84)
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.currentCellChanged.connect(lambda r, *_: self._show(r))
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        row.setContentsMargins(16, 0, 16, 0)
        row.setSpacing(8)
        add = button("Add images")
        add.clicked.connect(self.add_images)
        self.toggle_btn = button("Exclude")
        self.toggle_btn.clicked.connect(lambda: self._toggle(self._current))
        row.addWidget(add, 1)
        row.addWidget(self.toggle_btn, 1)
        v.addLayout(row)
        self.camera_link = QPushButton("Add a second camera")
        self.camera_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.camera_link.setStyleSheet(
            f'border: 0; background: transparent; padding: 0; min-height: 0; text-align: left; color: {T.INK_MUTED}; '
            f'font-family: "{T.MONO}"; font-size: 12px; font-weight: 500; text-decoration: underline;')
        self.camera_link.clicked.connect(self._toggle_second_camera)
        link_row = QHBoxLayout()
        link_row.setContentsMargins(16, 0, 16, 0)
        link_row.addWidget(self.camera_link)
        link_row.addStretch(1)
        v.addLayout(link_row)

        card = frame("sheet")
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.mouseReleaseEvent = lambda _e: open_site(where="side-card")
        ch = QHBoxLayout(card)
        ch.setContentsMargins(12, 12, 12, 12)
        ch.setSpacing(12)
        ch.addWidget(SvgMark("icon.svg", 36), 0, Qt.AlignmentFlag.AlignTop)
        cv_ = QVBoxLayout()
        cv_.setSpacing(4)
        cv_.addWidget(label("Calibration targets by Calib.Tools", font=T.sans(13, 600), wrap=True))
        cv_.addWidget(LinkLabel(T.VISIT, T.site_url(where="side-card"), "monolink"))
        ch.addLayout(cv_, 1)
        wrap = QHBoxLayout()
        wrap.setContentsMargins(16, 6, 16, 0)
        wrap.addWidget(card)
        v.addLayout(wrap)
        return w

    def _centre(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.view = ImageView()
        self.view.corrected_toggled.connect(lambda _on: self._show(self._current))
        self.view2 = ImageView()                 # the second camera's image of the same pair
        self.view2.corrected_btn.hide()          # one button, on the first view, switches both
        self.view_rule = rule("vrule")
        views = QHBoxLayout()
        views.setContentsMargins(0, 0, 0, 0)
        views.setSpacing(0)
        views.addWidget(self.view, 1)
        views.addWidget(self.view_rule)
        views.addWidget(self.view2, 1)
        v.addLayout(views, 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        v.addWidget(self.progress)
        v.addWidget(rule())
        bottom = QWidget()
        bottom.setFixedHeight(232)
        b = QHBoxLayout(bottom)
        b.setContentsMargins(20, 14, 20, 14)
        b.setSpacing(24)
        left = QVBoxLayout()
        left.setSpacing(6)
        self.chart_head = label("ERROR PER IMAGE", "eyebrow")
        left.addWidget(self.chart_head)
        self.chart = ErrorChart()
        self.chart.toggled.connect(self._toggle)
        self.chart.selected.connect(self._select_row)
        left.addWidget(self.chart, 1)
        b.addLayout(left, 1)
        right = QVBoxLayout()
        right.setSpacing(6)
        heads = QHBoxLayout()
        heads.setSpacing(12)
        self.map_tabs = []
        for text, tip in (("COVERAGE", "Where in the image the detected points fell"),
                          ("UNCERTAINTY", "How well the result is pinned down across the image"),
                          ("RESIDUALS", "What is left over after the fit, across the image")):
            b_ = QPushButton(text)
            b_.setCursor(Qt.CursorShape.PointingHandCursor)
            b_.setToolTip(tip)
            b_.setCheckable(True)
            b_.clicked.connect(lambda _on, which=len(self.map_tabs): self._show_map(which))
            heads.addWidget(b_)
            self.map_tabs.append(b_)
        heads.addStretch(1)
        right.addLayout(heads)
        self.coverage = CoverageMap()
        right.addWidget(self.coverage)
        self._show_map(0)
        right.addStretch(1)
        b.addLayout(right)
        v.addWidget(bottom)
        return w

    def _show_map(self, which: int) -> None:
        """Switch the small map between point coverage and projection uncertainty."""
        for i, b_ in enumerate(self.map_tabs):
            b_.setChecked(i == which)
            ink = T.INK if i == which else T.INK_MUTED
            b_.setStyleSheet(
                f'border: 0; border-bottom: 2px solid {T.INK if i == which else "transparent"}; '
                f'background: transparent; padding: 0 0 2px 0; min-height: 0; color: {ink}; '
                f'font-family: "{T.MONO}"; font-size: 12px; font-weight: 500;')
        self.coverage.set_mode(which == 1, residuals=which == 2)

    def _right(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(340)
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 6, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._tab_target(), "Target")
        self.tabs.addTab(self._tab_model(), "Model")
        self.tabs.addTab(self._tab_result(), "Result")
        v.addWidget(self.tabs)
        return w

    def _tab_target(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(12)
        v.addWidget(label("CALIB.TOOLS BOARD", "eyebrow"))
        self.preset = QComboBox()
        self.preset.setFixedHeight(36)
        self.preset.addItem("Custom target", None)
        for board in stock_boards():
            self.preset.addItem(board.part_number, board.part_number)
        self.preset.setMaxVisibleItems(18)
        self.preset.currentIndexChanged.connect(self._preset_chosen)
        v.addWidget(self.preset)
        self.preset_link = LinkLabel("", T.SITE)
        v.addWidget(self.preset_link)
        v.addWidget(rule())
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.kind = QComboBox()
        self.kind.addItem("ChArUco", CHARUCO)
        self.kind.addItem("Checkerboard", CHECKERBOARD)
        self.kind.addItem("Circle grid", CIRCLES)
        self.kind.addItem("Asymmetric circles", ASYMMETRIC_CIRCLES)
        self.kind.addItem("AprilGrid", APRILGRID)
        self.kind.addItem("ArUco grid", ARUCO_GRID)
        self._last_kind = CHARUCO
        self.sx, self.sy = QSpinBox(), QSpinBox()
        for s, val in ((self.sx, 11), (self.sy, 8)):
            s.setRange(2, 200)
            s.setValue(val)
        self.square, self.marker = QDoubleSpinBox(), QDoubleSpinBox()
        for s, val in ((self.square, 30.0), (self.marker, 22.0)):
            s.setRange(0.01, 5000)
            s.setDecimals(3)
            s.setSuffix(" mm")
            s.setValue(val)
        self.marker.setMinimum(0.0)  # 0 stands for "not known" where the field is optional (circle diameter)
        self.marker.setSpecialValueText("not set")
        self.first_id = QSpinBox()
        self.first_id.setRange(0, 9999)
        self.first_id.setToolTip("Boards in a set often continue the numbering: board 2 starts where board 1 ends.")
        self.first_id_label = label("First marker ID")
        self.dictionary = QComboBox()
        self.dictionary.addItems(DICTIONARIES)
        self.dictionary.setCurrentText("DICT_5X5_250")
        self.legacy = QCheckBox("Legacy pattern (older boards)")
        self.legacy.setToolTip("Tick this for ChArUco boards laid out the way OpenCV did before version 4.6.")
        self.marker_label = label("Marker size")
        self.dict_label = label("Dictionary")
        for wdg in (self.kind, self.sx, self.sy, self.square, self.marker, self.dictionary, self.first_id):
            wdg.setFixedHeight(36)
        for wdg in (self.sx, self.sy, self.square, self.marker, self.first_id):
            wdg.lineEdit().setTextMargins(7, 0, 0, 0)  # line the text up with the combo boxes
        form.addRow(label("Pattern"), self.kind)
        self.sx_label, self.sy_label, self.square_label = label(""), label(""), label("")
        form.addRow(self.sx_label, self.sx)
        form.addRow(self.sy_label, self.sy)
        form.addRow(self.square_label, self.square)
        form.addRow(self.marker_label, self.marker)
        form.addRow(self.dict_label, self.dictionary)
        form.addRow(self.first_id_label, self.first_id)
        v.addLayout(form)
        v.addWidget(self.legacy)
        self.target_hint = label("", "small", wrap=True)
        v.addWidget(self.target_hint)
        v.addStretch(1)
        v.addWidget(rule())
        self.target_note = label("", "small", wrap=True)
        v.addWidget(self.target_note)
        for wdg in (self.kind, self.dictionary):
            wdg.currentIndexChanged.connect(self._target_changed)
        for wdg in (self.sx, self.sy, self.square, self.marker, self.first_id):
            wdg.valueChanged.connect(self._target_changed)
        self.legacy.toggled.connect(self._target_changed)
        return w

    def _tab_model(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(6)
        v.addWidget(label("LENS MODEL", "eyebrow"))
        self.lens_kind = QComboBox()
        self.lens_kind.setFixedHeight(36)
        self.lens_kind.addItem("Standard lens", False)
        self.lens_kind.addItem("Fisheye lens", True)
        self.lens_kind.currentIndexChanged.connect(self._options_changed)
        v.addWidget(self.lens_kind)
        v.addSpacing(4)
        self.model_text = label("", "muted", wrap=True)
        v.addWidget(self.model_text)
        v.addSpacing(8)
        self.opt_tangential = QCheckBox("Tangential terms p1, p2")
        self.opt_tangential.setChecked(True)
        self.opt_k3 = QCheckBox("Third radial term k3")
        self.opt_rational = QCheckBox("Rational terms k4, k5, k6 (wide lenses)")
        self.opt_prism = QCheckBox("Thin prism terms s1 to s4")
        for c in (self.opt_tangential, self.opt_k3, self.opt_rational, self.opt_prism):
            v.addWidget(c)
            c.toggled.connect(self._options_changed)
        v.addSpacing(8)
        self.model_hint = label("", "small", wrap=True)
        v.addWidget(self.model_hint)
        v.addSpacing(16)
        v.addWidget(label("BOARD", "eyebrow"))
        self.opt_board = QCheckBox("Estimate the shape of the board")
        self.opt_board.toggled.connect(self._options_changed)
        v.addWidget(self.opt_board)
        v.addWidget(label("Leave this off for a flat, accurately made board. Tick it when the findings say the "
                          "images fit a board that is out of flat, or for a board printed and glued by hand. "
                          "The board's shape is then worked out together with the camera.", "small", wrap=True))
        v.addStretch(1)
        self._model_texts()
        return w

    def _model_texts(self) -> None:
        fisheye = bool(self.lens_kind.currentData())
        for c in (self.opt_tangential, self.opt_k3, self.opt_rational, self.opt_prism):
            c.setVisible(not fisheye)
        if fisheye:
            self.model_text.setText("OpenCV fisheye model: focal lengths, principal point, "
                                    "radial terms k1 to k4. For lenses wider than about 100 degrees.")
            self.model_hint.setText("A fisheye lens needs views that reach the corners of the image. "
                                    "Other software must read the result with its fisheye functions.")
        else:
            self.model_text.setText("OpenCV standard model: focal lengths, principal point, "
                                    "radial terms k1 and k2.")
            self.model_hint.setText("Start with the defaults. Add terms only if the error stays high "
                                    "at the image edges; more terms need more images.")

    def _tab_result(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(6)
        v.addWidget(label("RMS ERROR", "eyebrow"))
        self.rms_big = label("–", font=T.mono(34, 600))
        v.addWidget(self.rms_big)
        self.rms_note = label("", "small", wrap=True)
        v.addWidget(self.rms_note)
        self.used_note = label("", "muted", wrap=True)
        v.addWidget(self.used_note)
        v.addSpacing(8)
        # Two cameras: where the second sits, then a switch for whose own numbers follow.
        self.rig_box = QWidget()
        rv = QVBoxLayout(self.rig_box)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)
        rv.addWidget(label("CAMERA 2, SEEN FROM CAMERA 1", "eyebrow"))
        self.rig_keys = ("Apart", "Turned", "X (right), mm", "Y (down), mm", "Z (forward), mm")
        box = frame("sheet")
        g = QGridLayout(box)
        g.setContentsMargins(12, 6, 12, 6)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(2)
        self.rig_values = {}
        for i, k in enumerate(self.rig_keys):
            g.addWidget(label(k, font=T.mono(13), color=T.INK_MUTED), i, 0)
            val = label("–", font=T.mono(13, 600))
            g.addWidget(val, i, 1, Qt.AlignmentFlag.AlignRight)
            self.rig_values[k] = val
        g.setColumnStretch(1, 1)
        rv.addWidget(box)
        self.rig_note = label("", "small", wrap=True)
        self.rig_note.setToolTip(
            "X, Y and Z are where camera 2 sits in camera 1's own directions: to its right, downwards, and the way "
            "it looks. ± is one standard deviation from the scatter of the detected points and from how well each "
            "camera's own parameters are known.")
        rv.addWidget(self.rig_note)
        rv.addSpacing(8)
        tabs_row = QHBoxLayout()
        tabs_row.setSpacing(12)
        self.cam_tabs = []
        for i in range(2):
            b_ = QPushButton(f"CAMERA {i + 1}")
            b_.setCursor(Qt.CursorShape.PointingHandCursor)
            b_.clicked.connect(lambda _on, which=i: self._select_camera(which))
            tabs_row.addWidget(b_)
            self.cam_tabs.append(b_)
        tabs_row.addStretch(1)
        rv.addLayout(tabs_row)
        v.addWidget(self.rig_box)
        v.addWidget(label("INTRINSICS, PX", "eyebrow"))
        self.intr = self._kv_grid(("fx", "fy", "cx", "cy"))
        v.addWidget(self.intr[0])
        self.intr_note = label("", "small", wrap=True)
        self.intr_note.setToolTip(
            "One standard deviation, worked out from the scatter of the detected points. It does not cover "
            "a board that is not flat or not the size entered, or a lens model that does not fit the lens.")
        v.addWidget(self.intr_note)
        v.addSpacing(8)
        v.addWidget(label("DISTORTION", "eyebrow"))
        self.dist_box = frame("sheet")
        self.dist_layout = QGridLayout(self.dist_box)
        self.dist_layout.setContentsMargins(12, 6, 12, 6)
        self.dist_layout.setHorizontalSpacing(12)
        self.dist_layout.setVerticalSpacing(2)
        v.addWidget(self.dist_box)
        v.addSpacing(8)
        v.addWidget(label("FINDINGS", "eyebrow"))
        self.findings = QVBoxLayout()
        self.findings.setSpacing(6)
        v.addLayout(self.findings)
        v.addStretch(1)
        # Many lens terms and several findings do not fit a small window; the panel scrolls then.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        scroll.setWidget(w)
        return scroll

    def _kv_grid(self, keys):
        box = frame("sheet")
        g = QGridLayout(box)
        g.setContentsMargins(12, 6, 12, 6)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(2)
        values = {}
        for i, k in enumerate(keys):
            r, c = divmod(i, 2)
            g.addWidget(label(k, font=T.mono(13), color=T.INK_MUTED), r, c * 2)
            val = label("–", font=T.mono(13, 600))
            g.addWidget(val, r, c * 2 + 1, Qt.AlignmentFlag.AlignRight)
            values[k] = val
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        g.setColumnMinimumWidth(2, 28)
        return box, values

    def _footer(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(32)
        h = QHBoxLayout(w)
        h.setContentsMargins(20, 0, 20, 0)
        self.status = label("", font=T.mono(12), color=T.INK_MUTED)
        h.addWidget(self.status, 1)
        h.addWidget(label(f"OpenCV {cv2.__version__} · Plumbline {__version__} · early release ·",
                          font=T.mono(12), color=T.INK_MUTED))
        link = LinkLabel("calib.tools", T.site_url(where="footer"))
        link.setStyleSheet(f'font-family: "{T.MONO}"; font-size: 12px; color: {T.RED_TEXT};')
        h.addWidget(link)
        return w

    # ----------------------------------------------------------------- state
    def _preset_chosen(self, *_):
        """A part number was picked: fill in its pattern. 'Custom target' leaves the fields as they are."""
        board = find_board(self.preset.currentData())
        if board is not None:
            self._fill_fields(board.target)
        self._target_changed()

    def _fill_fields(self, t: Target) -> None:
        fields = (self.kind, self.sx, self.sy, self.square, self.marker, self.dictionary, self.legacy, self.first_id)
        for wdg in fields:
            wdg.blockSignals(True)
        self.kind.setCurrentIndex(max(0, self.kind.findData(t.kind)))
        self.sx.setValue(t.squares_x)
        self.sy.setValue(t.squares_y)
        self.square.setValue(t.square_mm)
        if t.is_circles:
            self.marker.setValue(t.marker_mm or 0.0)
        if t.kind in (CHARUCO, *MARKER_GRIDS):
            self.marker.setValue(t.marker_mm)
            self.dictionary.setCurrentText(t.dictionary)
            self.legacy.setChecked(t.legacy_pattern)
            self.first_id.setValue(t.first_marker_id)
        self._last_kind = t.kind
        for wdg in fields:
            wdg.blockSignals(False)

    def current_target(self) -> Target | None:
        kind = self.kind.currentData()
        try:
            if kind == CHARUCO:
                target = Target(CHARUCO, self.sx.value(), self.sy.value(), self.square.value(),
                                self.marker.value(), self.dictionary.currentText(), self.legacy.isChecked(),
                                first_marker_id=self.first_id.value())
            elif kind in MARKER_GRIDS:
                target = Target(kind, self.sx.value(), self.sy.value(), self.square.value(),
                                self.marker.value(), self.dictionary.currentText(),
                                first_marker_id=self.first_id.value())
            elif kind in (CIRCLES, ASYMMETRIC_CIRCLES):
                target = Target(kind, self.sx.value(), self.sy.value(), self.square.value(),
                                self.marker.value() or None)
            else:
                target = Target(kind, self.sx.value(), self.sy.value(), self.square.value())
            # The part number stays only while the fields still describe that board.
            board = find_board(self.preset.currentData())
            if board is not None and same_geometry(board.target, target):
                return board.target
            return target
        except ValueError as exc:
            self.target_hint.setText(str(exc).capitalize() + ".")
            self.target_hint.setStyleSheet(f"color: {T.RED_TEXT}; font-size: 12px;")
            return None

    def current_options(self) -> ModelOptions:
        return ModelOptions(
            k3=self.opt_k3.isChecked(), tangential=self.opt_tangential.isChecked(),
            rational=self.opt_rational.isChecked(), thin_prism=self.opt_prism.isChecked(),
            fisheye=bool(self.lens_kind.currentData()),
            release_board=self.opt_board.isChecked(),
        )

    def _target_changed(self, *_):
        kind = self.kind.currentData()
        charuco = kind == CHARUCO
        tags = kind in (CHARUCO, *MARKER_GRIDS)
        if kind != self._last_kind and tags:
            # ChArUco boards and AprilGrids use different marker families; start from the usual one.
            april = "APRILTAG" in self.dictionary.currentText()
            if (kind == APRILGRID) != april:
                self.dictionary.blockSignals(True)
                self.dictionary.setCurrentText("DICT_APRILTAG_36h11" if kind == APRILGRID else "DICT_5X5_250")
                self.dictionary.blockSignals(False)
        circles = kind in (CIRCLES, ASYMMETRIC_CIRCLES)
        if kind != self._last_kind:
            # The same field holds a marker size, a gap or a circle diameter; do not carry one over as another.
            was_circles = self._last_kind in (CIRCLES, ASYMMETRIC_CIRCLES)
            fresh = None
            if circles and not was_circles:
                fresh = 0.0
            elif kind == CHARUCO and self._last_kind != CHARUCO:
                fresh = round(0.75 * self.square.value(), 3)
            elif kind in MARKER_GRIDS and self._last_kind not in MARKER_GRIDS:
                fresh = round(0.3 * self.square.value(), 3)
            if fresh is not None:
                self.marker.blockSignals(True)
                self.marker.setValue(fresh)
                self.marker.blockSignals(False)
        self._last_kind = kind
        self.marker_label.setText({APRILGRID: "Tag gap", ARUCO_GRID: "Marker gap", CIRCLES: "Circle diameter",
                                   ASYMMETRIC_CIRCLES: "Circle diameter"}.get(kind, "Marker size"))
        self.first_id_label.setText("First tag ID" if kind == APRILGRID else "First marker ID")
        if kind == ARUCO_GRID:
            names = ("Markers across", "Markers down", "Marker size")
            note = ("Measure a marker across its black border, and the white gap to the next marker. "
                    "IDs run from the top-left marker to the right and then downward, as OpenCV's GridBoard "
                    "lays them out. Part of the grid in the image is enough.")
        elif kind == APRILGRID:
            names = ("Tags across", "Tags down", "Tag size")
            note = ("Measure a tag across its black border, and the white gap to the next tag. "
                    "IDs run from the bottom-left tag to the right and then upward, as Kalibr lays them out. "
                    "Part of the grid in the image is enough.")
        elif kind == CIRCLES:
            names = ("Circles across", "Circles down", "Spacing")
            note = ("For any other board: count the circles, and measure from the centre of one to the centre "
                    "of the next. With the diameter entered, the centres are corrected for perspective.")
        elif kind == ASYMMETRIC_CIRCLES:
            names = ("Columns", "Rows", "Lattice spacing")
            note = ("Columns and rows count the positions of the square lattice the circles sit on; "
                    "every other position holds a circle. The whole grid must be in every image. "
                    "With the diameter entered, the centres are corrected for perspective.")
        else:
            names = ("Squares across", "Squares down", "Square size")
            note = "For any other board: count squares, not inner corners, and measure one square."
        for wdg, text in zip((self.sx_label, self.sy_label, self.square_label), names):
            wdg.setText(text)
        self.target_note.setText(note)
        for wdg in (self.dictionary, self.dict_label, self.first_id, self.first_id_label):
            wdg.setVisible(tags)
        for wdg in (self.marker, self.marker_label):
            wdg.setVisible(tags or circles)
        self.legacy.setVisible(charuco)
        target = self.current_target()
        board = find_board(target.part_number) if target is not None else None
        if board is None and self.preset.currentIndex() != 0:
            self.preset.blockSignals(True)  # a field was edited away from the stock board
            self.preset.setCurrentIndex(0)
            self.preset.blockSignals(False)
        if board is not None:
            self.preset_link.setText(f"{board.part_number} on Calib.Tools \u2197")
            self.preset_link.set_url(T.site_url(board.product_path, "target-preset"))
        else:
            self.preset_link.setText("Find your board on Calib.Tools \u2197")
            self.preset_link.set_url(T.site_url("/targets/", "target-custom"))
        if target is not None:
            self.target_hint.setStyleSheet("")
            if target.kind in MARKER_GRIDS:
                one = "tag" if target.kind == APRILGRID else "marker"
                self.target_hint.setText(
                    f"{target.marker_count} {one}s, {target.corner_count} corners, "
                    f"pattern {target.width_mm:g} x {target.height_mm:g} mm, {one} IDs {target.first_marker_id} "
                    f"to {target.first_marker_id + target.marker_count - 1}."
                )
            elif target.is_circles:
                cols, rows = target.opencv_size
                self.target_hint.setText(
                    f"{target.corner_count} circles, {rows} rows of {cols}. "
                    f"OpenCV pattern size {cols} x {rows}."
                )
            else:
                self.target_hint.setText(
                    f"{target.inner_cols} x {target.inner_rows} inner corners, "
                    f"pattern {target.width_mm:g} x {target.height_mm:g} mm"
                    + (f", marker IDs {target.first_marker_id} to "
                       f"{target.first_marker_id + target.marker_count - 1}." if target.kind == CHARUCO else ".")
                )
            self.table.setHorizontalHeaderLabels(["File", target.point_word.capitalize(), "Error, px"])
        if target != self.project.target:
            self.project.target = target
            had = any(d.count for d in self.project.detections)
            self.project.detections = []
            self.project.excluded = set()
            self.result = None
            if had:
                self._say("Target changed. Run detection again.")
        self._refresh()

    def _options_changed(self, *_):
        self._model_texts()
        self.project.options = self.current_options()
        if self.result is not None:
            self.result = None
            self._say("Model changed. Calibrate again.")
        self._refresh()

    def load_project(self, project: Project) -> None:
        self.project = project
        self.result = None
        self._cam = 0
        self.view.corrected_btn.setChecked(False)  # every project opens on the images as they were taken
        t = project.target
        if t is not None:
            self._fill_fields(t)
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(max(0, self.preset.findData(t.part_number)) if t.part_number else 0)
            self.preset.blockSignals(False)
        o = project.options
        for box, val in ((self.opt_k3, o.k3), (self.opt_tangential, o.tangential),
                         (self.opt_rational, o.rational), (self.opt_prism, o.thin_prism),
                         (self.opt_board, o.release_board)):
            box.blockSignals(True)
            box.setChecked(val)
            box.blockSignals(False)
        self.lens_kind.blockSignals(True)
        self.lens_kind.setCurrentIndex(1 if o.fisheye else 0)
        self.lens_kind.blockSignals(False)
        self._model_texts()
        self._target_changed()
        if self.project.target is None:
            self.project.target = t
        self._refresh()
        if any(d.usable for d in self.project.detections):
            self.run_calibration()

    # ------------------------------------------------------- dropped files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        """Image files, or folders with images in them (one level of subfolders, for left/ and right/)."""
        found = []
        for url in e.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                inside = sorted(path.iterdir())
                for item in inside:
                    found.extend(sorted(item.iterdir()) if item.is_dir() else [item])
            else:
                found.append(path)
        images = [str(f) for f in found if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
        if images:
            e.acceptProposedAction()
            self.add_images(images)

    # --------------------------------------------------------------- cameras
    def _select_camera(self, which: int) -> None:
        if not self.project.second_camera or which == self._cam:
            return
        self._cam = which
        self._result = self._result_of(which)
        self._refresh()

    def _toggle_second_camera(self) -> None:
        p = self.project
        if self._busy():
            return
        if not p.second_camera:
            p.second_camera = True
            self._cam = 1
            self.result = None
            self._say("Camera 2 added. Add its images: they pair with camera 1's in file-name order.")
        else:
            if p.second_image_paths and QMessageBox.question(
                    self, "Plumbline", "Remove camera 2 and its images from this project?") \
                    != QMessageBox.StandardButton.Yes:
                return
            p.second_camera, p.second_image_paths, p.second_detections = False, [], []
            self._cam = 0
            self.result = None
            self._say("Camera 2 removed.")
        self._refresh()
        if not p.second_camera and any(d.usable for d in p.detections):
            self.run_calibration()

    def reset(self) -> None:
        self.load_project(Project(target=self.current_target()))
        self._say("")

    # --------------------------------------------------------------- actions
    def _goto_target(self):
        self.tabs.setCurrentIndex(0)

    def add_images(self, paths: list[str] | None = None):
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, "Add images", "", IMAGE_FILTER)
        pr = self.project
        known = set(pr.image_paths) | set(pr.second_image_paths)
        new = [p for p in paths if p not in known]
        if not new:
            return
        # Images of two cameras handed over together: the file names tell them apart and pair them.
        fresh = not pr.image_paths and not pr.second_image_paths
        level = pr.second_camera and len(pr.image_paths) == len(pr.second_image_paths)
        split = split_by_camera(new) if fresh or level else None
        if split is not None:
            pr.second_camera = True
            pr.image_paths.extend(split.first)
            pr.second_image_paths.extend(split.second)
            pr.detections, pr.second_detections = [], []
            said = (f"Two cameras found in the file names ({split.labels[0]} / {split.labels[1]}): "
                    f"{len(split.first)} pairs.")
            if split.alone:
                k = len(split.alone)
                self._left_out = (f" {k} image{'' if k == 1 else 's'} without a partner "
                                  f"{'was' if k == 1 else 'were'} left out.")
                said += self._left_out
        elif pr.second_camera:
            # Added to the selected camera. Pairs then go by file-name order within each camera.
            mine = pr.paths_of(self._cam)
            mine.extend(new)
            mine.sort(key=natural)
            pr.detections, pr.second_detections = [], []
            said = f"{len(new)} images added to camera {self._cam + 1}."
        else:
            pr.image_paths.extend(new)
            said = f"{len(new)} images added."
        self.result = None
        self._say(said)
        self._refresh()
        if self.project.target is not None:
            self.run_detection()

    def run_detection(self):
        target = self.project.target
        if target is None:
            self.tabs.setCurrentIndex(0)
            self._say("Fix the target settings first.")
            return
        if not self.project.paths_of(self._cam if self.project.second_camera else 0):
            self.add_images()
            return
        if self._busy():
            return
        paths = list(self.project.image_paths)
        paths2 = list(self.project.second_image_paths) if self.project.second_camera else []

        def job(progress):
            out = []
            for i, p in enumerate(paths + paths2):
                out.append(detect_file(p, target))
                progress(i + 1, len(paths) + len(paths2))
            return out

        def done(dets):
            self.project.detections = dets[:len(paths)]
            self.project.second_detections = dets[len(paths):]
            self.project.excluded = set()
            self.result = None
            good = sum(d.usable for d in dets)
            self._say(f"Detection finished: {good} of {len(dets)} images usable.")
            if not good:
                self.tabs.setCurrentIndex(0)  # nothing found: the target settings are the first suspect
            self._refresh()
            if good:
                self.run_calibration()

        self._start(job, done, "Detecting the target")

    def run_calibration(self):
        target = self.project.target
        dets = self.project.detections
        if target is None or not dets:
            self.run_detection()
            return
        if self._busy():
            return
        options, excluded = self.project.options, set(self.project.excluded)
        if self.project.second_camera:
            dets2 = self.project.second_detections
            if len(dets) != len(dets2):
                self.result = None
                self._say(f"Camera 1 has {len(dets)} images and camera 2 has {len(dets2)}. "
                          "Every pair needs one image from each camera, taken at the same moment.")
                self._refresh()
                return

            def done_rig(rig: RigResult):
                self.rig = rig
                self._result = rig.second if self._cam else rig.first
                self.project.last_rms_point = rig.rms_point
                self.tabs.setCurrentIndex(2)
                self._say(f"Calibrated 2 cameras from {len(rig.pairs)} pairs, RMS {rig.rms_point:.3f} px."
                          + self._left_out)
                self._left_out = ""
                self._refresh()

            self._start(lambda _p: calibrate_pair(dets, dets2, target, options, options, excluded), done_rig,
                        "Calibrating")
            return

        def done(result: CalibrationResult):
            self.result = result
            self.project.last_rms_point = result.rms_point
            self.tabs.setCurrentIndex(2)
            self._say(f"Calibrated from {len(result.views)} images, RMS {result.rms_point:.3f} px.")
            self._refresh()

        self._start(lambda _p: calibrate(dets, target, options, excluded), done, "Calibrating")

    def export_all(self):
        if self.project.second_camera:
            self._say("Exporting a project with two cameras is not in this version yet.")
            return
        if self.result is None:
            self._say("Calibrate first, then export.")
            return
        dialog = ExportDialog(self.project, self.result, self)
        if dialog.exec() != ExportDialog.DialogCode.Accepted:
            return
        self.export_to(dialog.folder, dialog.project_name, dialog.formats(), dialog.recheck_months)

    def export_to(self, folder: str, name: str | None = None, formats: set[str] | None = None,
                  recheck_months: int | None = None):
        """Write one revision: the chosen files plus the project file, all named after the project."""
        base = Path(folder)
        if name:
            self.project.title = safe_name(name)
        name = self.project.name
        formats = set(formats) if formats is not None else {"report", "opencv_yaml", "json", "ros"}
        if "example" in formats:
            formats.add("opencv_yaml")  # the example reads this file
        signature = result_signature(self.result)
        revision = self.project.revision_for(signature)
        files = export_files(name, revision)
        written = []
        if "opencv_yaml" in formats:
            export.write_opencv_yaml(self.result, base / files["opencv_yaml"])
            written.append(files["opencv_yaml"])
        if "opencv_xml" in formats:
            export.write_opencv_xml(self.result, base / files["opencv_xml"])
            written.append(files["opencv_xml"])
        if "json" in formats:
            export.write_json(self.result, base / files["json"])
            written.append(files["json"])
        if "ros" in formats and export.ros_distortion_model(self.result):
            export.write_ros_camera_info(self.result, base / files["ros"], safe_name(name))
            written.append(files["ros"])
        if "xlsx" in formats:
            write_xlsx(self.result, self.project.detections, base / files["xlsx"], name, revision)
            written.append(files["xlsx"])
        if "maps" in formats:
            write_maps(self.result, base / files["maps"])
            written.append(files["maps"])
        if "example" in formats:
            write_example(self.result, base / files["example"], files["opencv_yaml"],
                          files["maps"] if "maps" in formats else None)
            written.append(files["example"])
        if "report" in formats:
            write_report(self.result, self.project.detections, base / files["report"], name, revision)
            written.append(files["report"])
        written.append(files["project"])
        self.project.revision, self.project.export_signature = revision, signature
        self.project.exported_on = date.today().isoformat()
        if recheck_months is not None:
            self.project.recheck_months = recheck_months
        self.project.save(str(base / files["project"]))
        self.project_saved.emit(self.project.path, self.project.target.describe(), self.result.rms_point,
                                self.project.due_date())
        self._say(f"Revision {revision} saved to {base}: {', '.join(written)}.")
        self._refresh()

    def rename_project(self):
        text, ok = QInputDialog.getText(self, "Project name", "Name used for this project and its exported files:",
                                        text=self.project.name)
        if ok and text.strip():
            self.project.title = safe_name(text)
            self._refresh()

    def _toggle(self, index: int):
        if not self._row_usable(index):
            return
        self.project.excluded ^= {index}
        self.run_calibration()

    # --------------------------------------------------------------- helpers
    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _start(self, job, done, what: str):
        self._say(what + " …")
        self.progress.setRange(0, 0)
        self.progress.show()
        w = Worker(job, self)
        w.progress.connect(lambda a, b: (self.progress.setRange(0, b), self.progress.setValue(a)))
        w.finished_ok.connect(done)
        w.failed.connect(self._failed)
        w.finished.connect(self.progress.hide)
        self._worker = w
        w.start()

    def _failed(self, message: str):
        self.result = None
        self._say(message)
        self._refresh()
        QMessageBox.warning(self, "Plumbline", message)

    def _say(self, text: str):
        self.status.setText(text)

    def _select_row(self, index: int):
        if 0 <= index < self.table.rowCount():
            self.table.selectRow(index)

    def _view_for(self, index: int):
        if self.result is None:
            return None
        return next((v for v in self.result.views if v.index == index), None)

    def _row_usable(self, index: int) -> bool:
        """A row can be excluded or restored if at least one of its images shows the target."""
        cams = (0, 1) if self.project.second_camera else (0,)
        for cam in cams:
            dets = self.project.detections_of(cam)
            if 0 <= index < len(dets) and dets[index].usable:
                return True
        return False

    def _show(self, row: int):
        self._current = row
        two = self.project.second_camera
        self.view2.setVisible(two)
        self.view_rule.setVisible(two)
        if two:
            self._show_one(self.view, 0, row, self._result_of(0))
            self._show_one(self.view2, 1, row, self._result_of(1))
        else:
            self._show_one(self.view, 0, row, self.result)
        self._update_toggle()
        self._refresh_chart()

    def _show_one(self, view: ImageView, cam: int, row: int, r: CalibrationResult | None) -> None:
        """Put one camera's image of the selected row into a view."""
        two = self.project.second_camera
        paths, dets = self.project.paths_of(cam), self.project.detections_of(cam)
        if row < 0 or row >= len(paths):
            view.allow_corrected(False)
            if not two:
                view.set_message(self._empty_message())
            elif not paths:
                view.set_message(f"Camera {cam + 1}\n\nNo images yet.\n\nSelect this camera on the left\nand add its images.")
            else:
                view.set_message(f"Camera {cam + 1}\n\nNo image for this pair.")
            return
        det = dets[row] if row < len(dets) else None
        fit = next((v for v in r.views if v.index == row), None) if r is not None else None
        name = Path(paths[row]).name
        word = self.project.target.point_word if self.project.target else "corners"
        if two:
            # Half the width: the caption keeps to what tells the two images apart.
            name = f"Camera {cam + 1} · {name}"
            if det is None:
                caption = f"{name} · not checked yet"
            elif det.count == 0 or not det.usable:
                caption = f"{name} · {det.note}"
            elif fit is not None:
                caption = f"{name} · {det.count} {word} · {fit.rms_point:.3f} px"
            else:
                caption = f"{name} · {det.count} {word}"
        elif det is None:
            caption = f"{name} · not checked yet"
        elif det.count == 0 or not det.usable:
            caption = f"{name} · {det.note}"
        elif fit is not None:
            caption = (f"{name} · {det.width} × {det.height} · {det.count} {word} · "
                       f"{fit.rms_point:.3f} px · residuals ×{RESIDUAL_GAIN}")
        else:
            caption = f"{name} · {det.width} × {det.height} · {det.count} {word}"
        corners = det.corners if det is not None else np.zeros((0, 2))
        gray = read_gray(paths[row])
        residuals = fit.residuals if fit else None
        if view is self.view:
            view.allow_corrected(r is not None)
        if gray is not None and r is not None and self.view.corrected:
            gray, corners, _ = lens.corrected_view(gray, corners, r.camera_matrix, r.distortion, r.fisheye)
            residuals = None
            caption = caption.replace(f" · residuals ×{RESIDUAL_GAIN}", "") + " · corrected view"
        view.set_image(gray, corners, residuals, caption)
        # Ring the board points that are off the same way in every image, where this image shows them.
        if r is not None and r.points is not None and det is not None and len(corners):
            wanted = {int(r.points.ids[i]) for i in trusted_suspects(r)}
            if wanted:
                view.set_rings(np.asarray(corners)[[k for k, pid in enumerate(det.ids.tolist())
                                                    if pid in wanted]])

    def _update_toggle(self):
        self.toggle_btn.setEnabled(self._row_usable(self._current))
        self.toggle_btn.setText("Restore" if self._current in self.project.excluded else "Exclude")

    def _empty_message(self) -> str:
        if not self.project.image_paths:
            return ("Add 15 to 30 images of your target.\n\nMove the target, not only the camera: "
                    "tilt it up to about 35 degrees, and cover the corners and edges of the image.")
        return "Select an image on the left."

    def _refresh_chart(self):
        bars = []
        if self.project.second_camera:
            # One bar per pair: both images together. A pair that takes no part has no bar.
            by_index = {q.index: q.rms_point for q in self.rig.pairs} if self.rig is not None else {}
            for i in range(self.project.pair_count):
                bars.append((i, by_index.get(i), i in self.project.excluded, f"pair {i + 1:02d}"))
            self.chart.set_data(bars, self._current)
            return
        for i, p in enumerate(self.project.image_paths):
            v = self._view_for(i)
            err = v.rms_point if v else None
            if i in self.project.excluded and self.result is not None:
                err = self._excluded_error(i)
            bars.append((i, err, i in self.project.excluded, Path(p).name))
        self.chart.set_data(bars, self._current)

    def _excluded_error(self, index: int) -> float | None:
        return error_outside_fit(self.result, self.project.detections[index])

    def _refresh(self):
        p, r = self.project, self.result
        dets = p.detections
        n = len(p.image_paths)
        shown = p.name if len(p.name) <= 20 else p.name[:19] + "\u2026"
        full = shown + (f" \u00b7 r{p.revision}" if p.revision else "")
        # The header has room for about 20 characters here; a longer name is cut, never the layout.
        cut = full if len(full) <= 20 else full[:19] + "\u2026"
        self.name_btn.setText(cut)
        self.name_btn.setToolTip((full + "\n" if cut != full else "")
                                 + "Rename this project. Exported files carry this name.")
        two = p.second_camera
        self._lay_out_for(two)
        if two:
            self._fill_pairs()
            n = p.pair_count
        keep = self._current
        self.table.blockSignals(True)
        self.table.setRowCount(n)
        for i, path in enumerate(() if two else p.image_paths):
            det = dets[i] if i < len(dets) else None
            view = self._view_for(i)
            excluded = i in p.excluded
            corners = "" if det is None else str(det.count)
            if excluded:
                err = "out"
            elif view is not None:
                err = f"{view.rms_point:.3f}"
            elif det is not None and not det.usable:
                err = "no fit"
            else:
                err = ""
            for c, text in enumerate((Path(path).name, corners, err)):
                item = QTableWidgetItem(text)
                item.setToolTip(det.note if det is not None and det.note else path)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if excluded or (det is not None and not det.usable):
                    item.setForeground(Qt.GlobalColor.darkRed)
                self.table.setItem(i, c, item)
        self.table.blockSignals(False)
        if n and not 0 <= keep < n:
            keep = 0
        if n:
            self.table.selectRow(keep)
        self._show(keep if n else -1)
        if two:
            used = len(self.rig.pairs) if self.rig is not None else 0
            self.images_count.setText(f"{n} pairs · {used} used" if n else "")
        else:
            self.images_count.setText(f"{n} loaded" if n else "")

        # Coverage from the images that take part, of the camera on show.
        shown = p.detections_of(self._cam if two else 0)
        active = [d for i, d in enumerate(shown) if d.usable and i not in p.excluded]
        if active:
            grid, share = coverage_grid(active, active[0].width, active[0].height)
            self.coverage.set_grid(grid, share)
        else:
            share = 0.0
            self.coverage.set_grid(None)
        self._refresh_chart()
        res = self.result
        self.coverage.set_uncertainty(res.projection.sd if res is not None and res.projection is not None else None)
        self.coverage.set_residuals(res.pattern.mean if res is not None and res.pattern is not None else None)

        # Header steps.
        t = p.target
        if two:
            dets = list(p.detections) + list(p.second_detections)
            n_images = len(p.image_paths) + len(p.second_image_paths)
        good = sum(d.usable for d in dets)
        rig = self.rig
        if two:
            states = [
                ("done" if t else "next", t.short_label if t else "Check the settings"),
                ("done" if n_images else "next" if t else "off", f"2 cameras · {n} pairs" if n_images else "Add images"),
                ("done" if dets else "next" if n_images and t else "off",
                 f"{good} good · {len(dets) - good} to check" if dets else "Find the target"),
                ("done" if rig else "next" if good else "off", f"RMS {rig.rms_point:.2f} px" if rig else "Solve the cameras"),
                ("off", "Not in this version"),
            ]
        else:
            states = self._single_states(t, n, dets, good, r)
        for step, (kind, status) in zip(self.steps, states):
            step.set_state(kind, status)
        self._fill_rig()

        # Result tab.
        if r is None:
            self.rms_big.setText("–")
            self.rms_note.setText("")
            self.used_note.setText("No result yet.")
            for val in self.intr[1].values():
                val.setText("–")
            self.intr_note.setText("")
            self._fill_dist([])
            self._fill_findings([])
            return
        self.rms_big.setText(f"{r.rms_point:.3f} px")
        self.rms_note.setText(
            f"Per point, as OpenCV reports it. Counted per coordinate (as mrcal does) "
            f"the same result reads {r.rms_coordinate:.3f} px."
        )
        self.used_note.setText(
            f"{len(r.views)} of {n} images, {r.corner_total:,} {r.target.point_word} · "
            f"{len(r.excluded)} excluded · {len(r.skipped)} without a usable detection"
        )
        for k, val in zip(("fx", "fy", "cx", "cy"), (r.fx, r.fy, r.cx, r.cy)):
            self.intr[1][k].setText(f"{val:.2f}")
        u = r.uncertainty
        if u is None:
            self.intr_note.setText("")
        else:
            parts = [f"{k} {u.std_of(k):.2f}" for k in ("fx", "fy", "cx", "cy") if u.std_of(k) is not None]
            self.intr_note.setText("± one standard deviation, px: " + " · ".join(parts))
        self._fill_dist(list(zip(r.distortion_names, r.distortion)),
                        {k: u.std_of(k) for k in r.distortion_names} if u is not None else {})
        self._fill_findings(self._make_findings(r, share))
        if two and rig is not None:
            self.rms_big.setText(f"{rig.rms_point:.3f} px")
            self.rms_note.setText(f"Both cameras together, per point. Camera 1 alone {rig.first.rms_point:.3f} px, "
                                  f"camera 2 alone {rig.second.rms_point:.3f} px.")
            points = sum(q.points for q in rig.pairs)
            left_out = f" · {len(rig.unpaired)} not used" if rig.unpaired else ""
            self.used_note.setText(f"{len(rig.pairs)} of {n} pairs, {points:,} {r.target.point_word} in both "
                                   f"cameras' images{left_out}")

    @staticmethod
    def _single_states(t, n, dets, good, r):
        return [
            ("done" if t else "next", t.short_label if t else "Check the settings"),
            ("done" if n else "next" if t else "off", f"{n} loaded" if n else "Add images"),
            ("done" if dets else "next" if n and t else "off",
             f"{good} good · {len(dets) - good} to check" if dets else "Find the target"),
            ("done" if r else "next" if good else "off", f"RMS {r.rms_point:.2f} px" if r else "Solve the camera"),
            ("next" if r else "off", "Files and report"),
        ]

    def _lay_out_for(self, two: bool) -> None:
        """Show or hide what only a project with two cameras needs."""
        self.cams_box.setVisible(two)
        self.cams_rule.setVisible(two)
        self.rig_box.setVisible(two)
        self.images_head.setText("PAIRS" if two else "IMAGES")
        self.chart_head.setText("ERROR PER PAIR" if two else "ERROR PER IMAGE")
        self.chart.per = "pair" if two else "image"
        self.camera_link.setText("Remove camera 2" if two else "Add a second camera")
        hh = self.table.horizontalHeader()
        want = 4 if two else 3
        if self.table.columnCount() != want:
            self.table.setColumnCount(want)
            self.table.setHorizontalHeaderLabels(["Pair", "Cam 1", "Cam 2", "Error, px"] if two
                                                 else ["File", "Corners", "Error, px"])
            hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for c, width in (((1, 58), (2, 58), (3, 84)) if two else ((1, 70), (2, 84))):
                hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
                self.table.setColumnWidth(c, width)
        for i, (row_, sub, err) in enumerate(self.cam_rows):
            on = two and i == self._cam
            row_.setStyleSheet(f"QFrame {{ border: 1px solid {T.INK if on else T.HAIRLINE}; background: {T.SHEET}; }} "
                               "QLabel { border: 0; background: transparent; }")
            paths, dets = self.project.paths_of(i), self.project.detections_of(i)
            size = next((f"{d.width} × {d.height} · " for d in dets if d.width), "")
            sub.setText(f"{size}{len(paths)} images" if paths else "no images yet")
            res = self._result_of(i) if two else None
            err.setText(f"{res.rms_point:.3f} px" if res is not None else "")
        for i, b_ in enumerate(self.cam_tabs):
            on = i == self._cam
            b_.setStyleSheet(
                f'border: 0; border-bottom: 2px solid {T.INK if on else "transparent"}; background: transparent; '
                f'padding: 0 0 2px 0; min-height: 0; color: {T.INK if on else T.INK_MUTED}; '
                f'font-family: "{T.MONO}"; font-size: 12px; font-weight: 500;')

    def _fill_pairs(self) -> None:
        """The table with two cameras: one row per pair."""
        p = self.project
        by_index = {q.index: q for q in self.rig.pairs} if self.rig is not None else {}
        n = p.pair_count
        self.table.blockSignals(True)
        self.table.setRowCount(n)
        for i in range(n):
            counts, missing = [], False
            for cam in (0, 1):
                paths, dets = p.paths_of(cam), p.detections_of(cam)
                if i >= len(paths):
                    counts.append("–")
                    missing = True
                elif i >= len(dets):
                    counts.append("")
                else:
                    counts.append(str(dets[i].count))
                    missing = missing or not dets[i].usable
            excluded = i in p.excluded
            if excluded:
                err = "out"
            elif i in by_index:
                err = f"{by_index[i].rms_point:.3f}"
            elif missing and p.image_paths and p.second_image_paths and (p.detections or p.second_detections):
                err = "no pair"
            else:
                err = ""
            tip = " · ".join(Path(paths[i]).name for paths in (p.image_paths, p.second_image_paths) if i < len(paths))
            for c, text in enumerate((f"{i + 1:02d}", counts[0], counts[1], err)):
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if excluded or err == "no pair":
                    item.setForeground(Qt.GlobalColor.darkRed)
                self.table.setItem(i, c, item)
        self.table.blockSignals(False)

    def _fill_rig(self) -> None:
        rig = self.rig
        if rig is None:
            for val in self.rig_values.values():
                val.setText("–")
            self.rig_note.setText("Where camera 2 sits and how far it is turned, measured from camera 1.")
            return
        x, y, z = rig.position
        texts = (f"{rig.apart_mm:.2f} mm", f"{rig.turned_deg:.2f}°", f"{x:.2f}", f"{y:.2f}", f"{z:.2f}")
        for k, text in zip(self.rig_keys, texts):
            self.rig_values[k].setText(text)
        note = "Where camera 2 sits and how far it is turned, measured from camera 1."
        if np.isfinite(rig.apart_std_mm):
            note += f" ± one standard deviation: apart {rig.apart_std_mm:.2f} mm · turned {rig.turned_std_deg:.3f}°"
        self.rig_note.setText(note)

    @staticmethod
    def _clear(lay) -> None:
        """Empty a layout at once. deleteLater alone leaves the old widgets painted until the
        event loop gets round to them, and they show through the new ones."""
        while lay.count():
            item = lay.takeAt(0)
            wdg = item.widget()
            if wdg is not None:
                wdg.hide()
                wdg.setParent(None)
                wdg.deleteLater()
            elif item.layout() is not None:
                Workbench._clear(item.layout())

    def _fill_dist(self, pairs, spread=None):
        spread = spread or {}
        lay = self.dist_layout
        self._clear(lay)
        if not pairs:
            lay.addWidget(label("–", font=T.mono(13), color=T.INK_MUTED), 0, 0)
            return
        for i, (k, val) in enumerate(pairs):
            row, col = divmod(i, 2)
            lay.addWidget(label(k, font=T.mono(13), color=T.INK_MUTED), row, col * 2)
            cell = label(f"{val:.5f}", font=T.mono(13, 600))
            if spread.get(k) is not None:
                cell.setToolTip(f"{k} = {val:.6f} ± {spread[k]:.6f} (one standard deviation)")
            lay.addWidget(cell, row, col * 2 + 1, Qt.AlignmentFlag.AlignRight)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)
        lay.setColumnMinimumWidth(2, 28)

    def _make_findings(self, r: CalibrationResult, share: float):
        colors = {PASS: T.GREEN_TEXT, CHECK: T.RED_TEXT, NOTE: T.INK_MUTED}
        return [(tag, colors[tag], text) for tag, text in findings(r, share)]

    def _fill_findings(self, items):
        lay = self.findings
        self._clear(lay)
        for tag, color, text in items:
            row = QHBoxLayout()
            row.setSpacing(10)
            t = label(tag, font=T.mono(11, 600), color=color)
            t.setFixedWidth(44)
            row.addWidget(t, 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(label(text, "muted", wrap=True), 1)
            lay.addLayout(row)
