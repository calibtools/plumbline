"""Start window: the first thing a user sees after installing."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from .. import __version__
from . import theme as T
from .widgets import LinkLabel, SvgMark, button, frame, label, open_site, rule

VERSION_LINE = "Version 0.1 · early release"


def recent_projects() -> list[dict]:
    items = QSettings().value("recent", []) or []
    return [i for i in items if isinstance(i, dict) and Path(i.get("path", "")).exists()][:5]


def remember_project(path: str, target: str, rms: float | None) -> None:
    from datetime import date

    s = QSettings()
    items = [i for i in (s.value("recent", []) or []) if isinstance(i, dict) and i.get("path") != path]
    items.insert(0, {"path": path, "target": target, "rms": rms, "date": date.today().isoformat()})
    s.setValue("recent", items[:8])


class RecentRow(QFrame):
    clicked = Signal(str)

    def __init__(self, item: dict):
        super().__init__()
        self._path = item["path"]
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(48)
        self.setStyleSheet(f"RecentRow {{ border: 0; border-top: 1px solid {T.HAIRLINE}; }} RecentRow:hover {{ background: {T.PAPER}; }}")
        g = QGridLayout(self)
        g.setContentsMargins(16, 0, 16, 0)
        g.setHorizontalSpacing(12)
        rms = item.get("rms")
        cells = [
            (label(Path(self._path).stem, font=T.mono(13, 600)), Qt.AlignmentFlag.AlignLeft),
            (label(item.get("target", ""), font=T.mono(13), color=T.INK_MUTED), Qt.AlignmentFlag.AlignLeft),
            (label(f"{rms:.2f}" if isinstance(rms, (int, float)) else "–", font=T.mono(13)), Qt.AlignmentFlag.AlignRight),
            (label(item.get("date", ""), font=T.mono(13), color=T.INK_MUTED), Qt.AlignmentFlag.AlignRight),
        ]
        for c, ((w, align), stretch) in enumerate(zip(cells, RECENT_STRETCH)):
            g.addWidget(w, 0, c, align | Qt.AlignmentFlag.AlignVCenter)
            g.setColumnStretch(c, stretch)

    def mouseReleaseEvent(self, _e):
        self.clicked.emit(self._path)


RECENT_STRETCH = (16, 16, 8, 10)


class WelcomePage(QWidget):
    new_project = Signal()
    open_project = Signal()
    open_recent = Signal(str)
    try_sample = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._left())
        body.addWidget(rule("vrule"))
        body.addWidget(self._right(), 1)
        outer.addLayout(body, 1)
        outer.addWidget(rule("inkrule"))
        outer.addWidget(self._footer())

    # -- left column -------------------------------------------------------
    def _left(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(520)
        v = QVBoxLayout(w)
        v.setContentsMargins(48, 40, 48, 24)
        v.setSpacing(0)

        v.addWidget(SvgMark("wordmark.svg", 64))
        v.addSpacing(20)
        v.addWidget(rule())
        v.addSpacing(16)
        name_row = QHBoxLayout()
        name_row.setSpacing(12)
        name_row.addWidget(label("PLUMBLINE", font=T.mono(22, 600)), 0, Qt.AlignmentFlag.AlignBottom)
        name_row.addWidget(label(VERSION_LINE, font=T.mono(12), color=T.INK_MUTED), 0, Qt.AlignmentFlag.AlignBottom)
        name_row.addStretch(1)
        v.addLayout(name_row)
        v.addSpacing(6)
        v.addWidget(label("Camera calibration software by Calib.Tools", font=T.sans(16)))
        v.addSpacing(24)

        b_new = button("New project", "bigprimary")
        b_open = button("Open project", "big")
        b_sample = button("Try with sample images", "big")
        b_new.clicked.connect(self.new_project)
        b_open.clicked.connect(self.open_project)
        b_sample.clicked.connect(self.try_sample)
        for b in (b_new, b_open, b_sample):
            v.addWidget(b)
            v.addSpacing(10)
        v.addWidget(label("24 rendered images of a ChArUco board. No camera or board needed.", "small"))
        v.addSpacing(24)

        v.addWidget(label("IN THIS VERSION", "eyebrow"))
        v.addSpacing(8)
        box = frame("sheet")
        g = QGridLayout(box)
        g.setContentsMargins(12, 4, 12, 4)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(0)
        rows = [
            ("Cameras", "One at a time"),
            ("Patterns", "ChArUco, checkerboard"),
            ("Lens model", "OpenCV standard"),
            ("Export", "OpenCV YAML and JSON, ROS camera_info"),
        ]
        for r, (k, val) in enumerate(rows):
            a, b = label(k, "muted"), label(val)
            a.setFixedHeight(32)
            g.addWidget(a, r, 0)
            g.addWidget(b, r, 1)
        g.setColumnMinimumWidth(0, 96)
        g.setColumnStretch(1, 1)
        v.addWidget(box)
        v.addSpacing(8)
        v.addWidget(label("Next: PDF report, more patterns, fisheye lenses, uncertainty, several cameras.", "small", wrap=True))

        v.addStretch(1)
        v.addWidget(label("Your images stay on this computer. Plumbline sends nothing.", "muted", wrap=True))
        return w

    # -- right column ------------------------------------------------------
    def _right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(48, 40, 48, 24)
        v.setSpacing(0)

        head = QHBoxLayout()
        head.setSpacing(12)
        bar = QFrame()
        bar.setFixedSize(40, 3)
        bar.setStyleSheet(f"background: {T.RED};")
        head.addWidget(bar, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(label("RECENT PROJECTS", "eyebrow"))
        head.addStretch(1)
        v.addLayout(head)
        v.addSpacing(10)

        self._recent_box = frame("sheet")
        self._recent_layout = QVBoxLayout(self._recent_box)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(0)
        v.addWidget(self._recent_box)
        v.addSpacing(28)

        v.addWidget(label("BEFORE YOU START", "eyebrow"))
        v.addSpacing(10)
        cards = QHBoxLayout()
        cards.setSpacing(12)
        data = [
            ("No board yet?", "Print a ChArUco or checkerboard pattern. Fine for a first try.",
             "Open the pattern generator", T.SITE + "/tools/"),
            ("Which size?", "The board should fill most of the frame at your working distance.",
             "Check size for your lens", T.SITE + "/tools/"),
            ("Have a Calib.Tools board?", "Enter its square count, sizes and dictionary from the label or data sheet in step 1.",
             T.VISIT, T.SITE),
        ]
        for title, text, link, url in data:
            c = frame("sheet")
            c.setMinimumHeight(150)
            cv = QVBoxLayout(c)
            cv.setContentsMargins(16, 16, 16, 16)
            cv.setSpacing(8)
            cv.addWidget(label(title, font=T.sans(15, 600), wrap=True))
            cv.addWidget(label(text, "muted", wrap=True))
            cv.addStretch(1)
            cv.addWidget(LinkLabel(link, url))
            cards.addWidget(c, 1)
        v.addLayout(cards)
        v.addStretch(1)

        band = frame("inksheet")
        band.setCursor(Qt.CursorShape.PointingHandCursor)
        band.mouseReleaseEvent = lambda _e: open_site()
        bh = QHBoxLayout(band)
        bh.setContentsMargins(20, 20, 20, 20)
        bh.setSpacing(16)
        bh.addWidget(SvgMark("icon.svg", 56))
        txt = QVBoxLayout()
        txt.setSpacing(4)
        txt.addWidget(label("Plumbline is made by Calib.Tools", font=T.sans(18, 600)))
        txt.addWidget(label("We make calibration targets on composite, ceramic and glass, "
                            "and the software tools for the work around them.", "muted", wrap=True))
        bh.addLayout(txt, 1)
        visit = button(T.VISIT, "ink")
        visit.clicked.connect(lambda: open_site())
        visit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        bh.addWidget(visit, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addWidget(band)
        v.addSpacing(20)

        links = QHBoxLayout()
        links.setSpacing(24)
        links.addWidget(LinkLabel(T.VISIT, T.SITE, "monolink"))
        links.addWidget(LinkLabel("User guide", T.SITE + "/guides/"))
        links.addWidget(LinkLabel("Report a problem", "https://github.com/calibtools/plumbline/issues"))
        links.addWidget(LinkLabel("Source code, MIT license", "https://github.com/calibtools/plumbline"))
        links.addStretch(1)
        v.addLayout(links)
        self.refresh_recent()
        return w

    def refresh_recent(self) -> None:
        lay = self._recent_layout
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        head = QFrame()
        head.setFixedHeight(32)
        head.setStyleSheet(f"background: {T.RED_TINT}; border: 0;")
        g = QGridLayout(head)
        g.setContentsMargins(16, 0, 16, 0)
        g.setHorizontalSpacing(12)
        aligns = (Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignRight)
        for c, (text, stretch) in enumerate(zip(("Project", "Target", "RMS, px", "Last opened"), RECENT_STRETCH)):
            g.addWidget(label(text, font=T.mono(12, 500), color=T.RED_TEXT), 0, c, aligns[c] | Qt.AlignmentFlag.AlignVCenter)
            g.setColumnStretch(c, stretch)
        lay.addWidget(head)
        items = recent_projects()
        if not items:
            empty = label("No projects yet. Start with New project or the sample images.", "muted")
            empty.setContentsMargins(16, 16, 16, 16)
            lay.addWidget(empty)
            return
        for item in items:
            row = RecentRow(item)
            row.clicked.connect(self.open_recent)
            lay.addWidget(row)

    def _footer(self) -> QWidget:
        import cv2

        w = QWidget()
        w.setFixedHeight(32)
        h = QHBoxLayout(w)
        h.setContentsMargins(20, 0, 20, 0)
        h.addWidget(label(f"Built on OpenCV {cv2.__version__}", font=T.mono(12), color=T.INK_MUTED))
        h.addStretch(1)
        h.addWidget(label(f"Plumbline {__version__} · early release ·", font=T.mono(12), color=T.INK_MUTED))
        link = LinkLabel("calib.tools", T.SITE)
        link.setStyleSheet(f'font-family: "{T.MONO}"; font-size: 12px; color: {T.RED_TEXT};')
        h.addWidget(link)
        return w
