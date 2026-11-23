"""About Plumbline: which version runs on what, and where to report a problem."""
from __future__ import annotations

import platform
import sys
from urllib.parse import quote

import cv2
import numpy as np
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QUrl, Qt, qVersion
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout

from .. import __version__
from . import theme as T
from .widgets import LinkLabel, button, frame, label, rule

REPORT_TO = "hello@calib.tools"
ISSUES = "https://github.com/calibtools/plumbline/issues"


def details() -> str:
    """What a report needs to say about the program and the computer. Nothing about the user or their images."""
    system = platform.system()
    if system == "Darwin":
        os_line = f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    elif system == "Windows":
        os_line = f"Windows {platform.release()} {platform.version()} ({platform.machine()})"
    else:
        os_line = f"{system} {platform.release()} ({platform.machine()})"
    return "\n".join((
        f"Plumbline {__version__}",
        f"OpenCV {cv2.__version__}",
        f"Qt {qVersion()} (PySide6 {pyside_version})",
        f"NumPy {np.__version__}",
        f"Python {sys.version.split()[0]}",
        os_line,
    ))


def report_url() -> str:
    """An e-mail to Calib.Tools, started with the details; the user writes what happened and sends it themselves."""
    body = ("What I did:\n\n\nWhat I expected:\n\n\nWhat happened instead:\n\n\n"
            "(Images or the project file help. Attach them if you can share them.)\n\n---\n" + details() + "\n")
    return f"mailto:{REPORT_TO}?subject={quote('Plumbline ' + __version__ + ': problem report')}&body={quote(body)}"


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Plumbline")
        self.setModal(True)
        self.setFixedWidth(520)
        self.setStyleSheet(f"QDialog {{ background: {T.PAPER}; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(12)
        v.addWidget(label("PLUMBLINE", "eyebrow"))
        v.addWidget(label("Camera calibration software by Calib.Tools", font=T.sans(18, 600), color=T.INK))
        v.addWidget(label("Early release. Results have been checked with rendered images so far, not yet with "
                          "photographs of real boards. Compare with a method you trust before relying on it.",
                          "small", wrap=True))
        box = frame("sheet")
        bv = QVBoxLayout(box)
        bv.setContentsMargins(14, 10, 14, 10)
        self.text = label(details(), font=T.mono(12), color=T.INK)
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bv.addWidget(self.text)
        v.addWidget(box)
        v.addWidget(label("Reporting a problem: these lines say which version runs on what. They hold nothing about "
                          "you or your images. Plumbline sends nothing by itself.", "small", wrap=True))
        row = QHBoxLayout()
        row.setSpacing(8)
        self.copy = button("Copy details")
        self.copy.clicked.connect(self._copy)
        self.report = button("Report a problem", "primary")
        self.report.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(report_url())))
        row.addWidget(self.copy)
        row.addWidget(self.report)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(rule())
        links = QHBoxLayout()
        links.setSpacing(18)
        links.addWidget(LinkLabel(T.VISIT, T.site_url(where="about")))
        links.addWidget(LinkLabel("Known issues on GitHub ↗", ISSUES))
        links.addStretch(1)
        links.addWidget(label("MIT license", "small"))
        v.addLayout(links)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(details())
        self.copy.setText("Copied")
