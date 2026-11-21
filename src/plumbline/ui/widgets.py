"""Shared widgets: brand marks, labels, charts and the image viewer."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QSizePolicy, QWidget

from . import theme as T


def open_site(path: str = "") -> None:
    QDesktopServices.openUrl(QUrl(T.SITE + path))


class SvgMark(QWidget):
    """An SVG drawn at a fixed height, width following its aspect ratio. Click opens calib.tools."""

    def __init__(self, name: str, height: int, clickable: bool = True, parent=None):
        super().__init__(parent)
        self._r = QSvgRenderer(T.asset(name))
        size = self._r.defaultSize()
        self.setFixedSize(QSize(round(height * size.width() / size.height()), height))
        if clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip("calib.tools")
        self._clickable = clickable

    def paintEvent(self, _e):
        p = QPainter(self)
        self._r.render(p, QRectF(self.rect()))

    def mouseReleaseEvent(self, _e):
        if self._clickable:
            open_site()


def label(text: str = "", role: str | None = None, font=None, wrap: bool = False, color: str | None = None) -> QLabel:
    lb = QLabel(text)
    if role:
        lb.setProperty("role", role)
    if font is not None:
        lb.setFont(font)
        # A font set in code must win over the style sheet's default size.
        lb.setStyleSheet(
            f'font-family: "{font.family()}"; font-size: {font.pixelSize()}px; '
            f"font-weight: {int(font.weight())};" + (f" color: {color};" if color else "")
        )
    elif color:
        lb.setStyleSheet(f"color: {color};")
    lb.setWordWrap(wrap)
    return lb


class LinkLabel(QLabel):
    """Text that opens a page on calib.tools (or any URL) when clicked."""

    def __init__(self, text: str, url: str = T.SITE, role: str = "link", parent=None):
        super().__init__(text, parent)
        self._url = url
        self.setProperty("role", role)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(url)

    def mouseReleaseEvent(self, _e):
        QDesktopServices.openUrl(QUrl(self._url))


def button(text: str, kind: str | None = None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def rule(role: str = "rule") -> QFrame:
    f = QFrame()
    f.setProperty("role", role)
    return f


def frame(role: str = "sheet") -> QFrame:
    f = QFrame()
    f.setProperty("role", role)
    return f


class Worker(QThread):
    """Runs fn(progress) off the UI thread. progress(done, total) may be called from fn."""

    progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.finished_ok.emit(self._fn(self.progress.emit))
        except Exception as exc:  # shown to the user by the caller
            self.failed.emit(str(exc))


class ErrorChart(QWidget):
    """Reprojection error per image. Click a bar to exclude or bring back that image."""

    toggled = Signal(int)
    selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._bars: list[tuple[int, float | None, bool, str]] = []  # index, error, excluded, name
        self._rects: list[tuple[QRectF, int]] = []
        self._current = -1

    def set_data(self, bars, current: int = -1):
        self._bars = bars
        self._current = current
        self.update()

    def sizeHint(self):
        return QSize(600, 180)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()
        left, right, top, bottom = 44, 8, 10, 22
        plot = QRectF(left, top, w - left - right, h - top - bottom)
        p.setFont(T.mono(11))
        self._rects = []
        errs = [b[1] for b in self._bars if b[1] is not None]
        if not errs:
            p.setPen(QColor(T.INK_SOFT))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Errors appear here after calibration")
            return
        used = [b[1] for b in self._bars if b[1] is not None and not b[2]]
        # Scale to the images in use so that one bad image does not flatten the rest.
        top_val = max(0.1, 1.25 * max(used or errs))
        ticks = _nice_ticks(top_val)
        for t in ticks:
            y = plot.bottom() - plot.height() * t / top_val
            p.setPen(QPen(QColor(T.HAIRLINE), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(T.INK_MUTED))
            p.drawText(QRectF(0, y - 8, left - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{t:g}")
        n = len(self._bars)
        slot = plot.width() / n
        bw = max(3.0, min(22.0, slot * 0.62))
        for k, (idx, err, excluded, _name) in enumerate(self._bars):
            x = plot.left() + slot * k + (slot - bw) / 2
            if err is None:
                continue
            clipped = err > top_val
            bh = plot.height() * min(err, top_val) / top_val
            r = QRectF(x, plot.bottom() - bh, bw, bh)
            if excluded:
                p.setPen(QPen(QColor(T.RED_TEXT), 1.5))
                p.setBrush(QColor(T.SHEET))
            else:
                p.setPen(QPen(QColor(T.INK), 1))
                p.setBrush(QColor(T.RED if idx == self._current else T.INK))
            p.drawRect(r.adjusted(0.5, 0.5, -0.5, 0))
            if clipped:  # the bar runs off the scale: mark it and print the value
                p.setPen(QColor(T.RED_TEXT))
                p.drawText(QRectF(x - 20, plot.top() - 10, bw + 40, 12), Qt.AlignmentFlag.AlignCenter, f"{err:.2f}")
            self._rects.append((QRectF(plot.left() + slot * k, plot.top(), slot, plot.height()), idx))
        p.setPen(QPen(QColor(T.INK), 1))
        p.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        p.setPen(QColor(T.INK_MUTED))
        p.drawText(QRectF(plot.left(), h - 18, plot.width(), 16), Qt.AlignmentFlag.AlignLeft,
                   "px per image, RMS per point · click a bar to exclude or restore it")

    def _hit(self, pos) -> int | None:
        for r, idx in self._rects:
            if r.contains(pos):
                return idx
        return None

    def mouseMoveEvent(self, e):
        idx = self._hit(e.position())
        if idx is None:
            self.setToolTip("")
            return
        for i, err, excluded, name in self._bars:
            if i == idx and err is not None:
                self.setToolTip(f"{name}  {err:.3f} px" + ("  (excluded)" if excluded else ""))

    def mouseReleaseEvent(self, e):
        idx = self._hit(e.position())
        if idx is None:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(idx)
            self.toggled.emit(idx)


def _nice_ticks(top: float) -> list[float]:
    raw = top / 4
    mag = 10 ** np.floor(np.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda s: abs(s * mag - raw)) * mag
    return [round(step * i, 6) for i in range(0, int(top / step) + 1)]


class CoverageMap(QWidget):
    """Where in the image the corners fell. Empty cells are outlined in red."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(244, 170)
        self._grid: np.ndarray | None = None
        self._share = 0.0

    def set_grid(self, grid: np.ndarray | None, share: float = 0.0):
        self._grid, self._share = grid, share
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setFont(T.mono(11))
        area = QRectF(0, 0, 240, 150)
        if self._grid is None:
            p.setPen(QPen(QColor(T.HAIRLINE), 1))
            p.drawRect(area.adjusted(0.5, 0.5, -0.5, -0.5))
            p.setPen(QColor(T.INK_SOFT))
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "No corners yet")
            return
        rows, cols = self._grid.shape
        cw, ch = area.width() / cols, area.height() / rows
        positive = self._grid[self._grid > 0]
        hi = float(np.percentile(positive, 75)) if positive.size else 1.0
        for r in range(rows):
            for c in range(cols):
                v = int(self._grid[r, c])
                cell = QRectF(c * cw + 1, r * ch + 1, cw - 2, ch - 2)
                if v == 0:
                    p.setPen(QPen(QColor(T.RED_TEXT), 1))
                    p.setBrush(QColor(T.SHEET))
                else:
                    shade = T.INK if v >= hi else T.LINE_STRONG if v >= hi / 3 else T.HAIRLINE
                    p.setPen(QPen(QColor(shade), 1))
                    p.setBrush(QColor(shade))
                p.drawRect(cell)
        p.setPen(QColor(T.INK_MUTED))
        p.drawText(QRectF(0, 154, 244, 16), Qt.AlignmentFlag.AlignLeft,
                   f"{self._share * 100:.0f} % of the image has corners")


RESIDUAL_GAIN = 30  # residual vectors are drawn this many times longer than they are


class ImageView(QWidget):
    """Shows one image with its detected corners and, after calibration, the residuals."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self._pix: QPixmap | None = None
        self._corners = np.zeros((0, 2))
        self._residuals: np.ndarray | None = None
        self._message = ""
        self._caption = ""

    def set_message(self, text: str):
        self._pix, self._message, self._caption = None, text, ""
        self.update()

    def set_image(self, gray: np.ndarray | None, corners: np.ndarray, residuals: np.ndarray | None, caption: str):
        if gray is None:
            self.set_message("This image could not be read")
            return
        h, w = gray.shape[:2]
        img = QImage(np.ascontiguousarray(gray).data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        self._pix = QPixmap.fromImage(img)
        self._corners, self._residuals, self._caption = corners, residuals, caption
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(T.PAPER_ALT))
        if self._pix is None:
            p.setPen(QColor(T.INK_MUTED))
            p.setFont(T.sans(15))
            p.drawText(self.rect().adjusted(40, 0, -40, 0),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._message)
            return
        pad, cap = 16, 44  # room for a two-line caption
        avail = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad - cap)
        s = min(avail.width() / self._pix.width(), avail.height() / self._pix.height())
        w, h = self._pix.width() * s, self._pix.height() * s
        x0, y0 = avail.left() + (avail.width() - w) / 2, avail.top() + (avail.height() - h) / 2
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawPixmap(QRectF(x0, y0, w, h), self._pix, QRectF(self._pix.rect()))
        p.setPen(QPen(QColor(T.INK), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(x0, y0, w, h))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for k, (cx, cy) in enumerate(self._corners):
            pt = QPointF(x0 + (cx + 0.5) * s, y0 + (cy + 0.5) * s)
            p.setPen(QPen(QColor("#FFFFFF"), 1))
            p.setBrush(QColor(T.RED))
            p.drawEllipse(pt, 2.6, 2.6)
            if self._residuals is not None and k < len(self._residuals):
                dx, dy = self._residuals[k]
                p.setPen(QPen(QColor(T.RED), 1.4))
                p.drawLine(pt, QPointF(pt.x() + dx * RESIDUAL_GAIN * s, pt.y() + dy * RESIDUAL_GAIN * s))
        p.setPen(QColor(T.INK_MUTED))
        p.setFont(T.mono(12))
        p.drawText(QRectF(pad, self.height() - pad - 34, self.width() - 2 * pad, 36),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, self._caption)
