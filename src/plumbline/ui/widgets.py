"""Shared widgets: brand marks, labels, charts and the image viewer."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QSizePolicy, QWidget

from . import theme as T


def open_site(path: str = "", where: str = "app") -> None:
    QDesktopServices.openUrl(QUrl(T.site_url(path, where)))


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
            open_site(where="logo")


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
        self.set_url(url)
        self.setProperty("role", role)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_url(self, url: str) -> None:
        self._url = url
        self.setToolTip(url.split("?")[0])

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
                   "px per image · click a bar to exclude or restore it")

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
        self._spread: np.ndarray | None = None
        self._lean: np.ndarray | None = None
        self.show_uncertainty = False
        self.show_residuals = False

    def set_grid(self, grid: np.ndarray | None, share: float = 0.0):
        self._grid, self._share = grid, share
        self.update()

    def set_uncertainty(self, sd: np.ndarray | None):
        """Projection uncertainty per cell in px (one standard deviation), or None."""
        self._spread = sd
        self.update()

    def set_residuals(self, mean: np.ndarray | None):
        """Average of observed minus predicted per cell, (rows, cols, 2) in px, or None."""
        self._lean = mean
        self.update()

    def _paint_residuals(self, p: QPainter, area: QRectF):
        lean = self._lean
        p.setPen(QPen(QColor(T.HAIRLINE), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(area.adjusted(0.5, 0.5, -0.5, -0.5))
        if lean is None:
            p.setPen(QColor(T.INK_SOFT))
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "No result yet")
            return
        rows, cols = lean.shape[:2]
        cw, ch = area.width() / cols, area.height() / rows
        size = np.linalg.norm(lean, axis=2)
        top = float(np.nanmax(size)) if np.isfinite(size).any() else 0.0
        # Lines are drawn so that a typical large one spans a cell; a few much larger ones are capped,
        # otherwise they would shrink everything else to dots. With almost nothing left over, lines stay short.
        finite = size[np.isfinite(size)]
        scale = float(np.percentile(finite, 85)) if finite.size else 0.0
        reach = 1.6 * min(cw, ch)
        gain = min(cw, ch) / max(scale, 0.05)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for r in range(rows):
            for c in range(cols):
                if not np.isfinite(size[r, c]):
                    continue
                mid = QPointF((c + 0.5) * cw, (r + 0.5) * ch)
                dx, dy = lean[r, c] * gain
                length = float(np.hypot(dx, dy))
                if length > reach:
                    dx, dy = dx * reach / length, dy * reach / length
                strong = size[r, c] >= 0.1
                p.setPen(QPen(QColor(T.RED if strong else T.INK), 1.4 if strong else 1.0))
                p.setBrush(QColor(T.RED if strong else T.INK))
                p.drawLine(QPointF(mid.x() - dx / 2, mid.y() - dy / 2), QPointF(mid.x() + dx / 2, mid.y() + dy / 2))
                p.drawEllipse(QPointF(mid.x() + dx / 2, mid.y() + dy / 2), 1.5, 1.5)   # the dot marks the far end
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QColor(T.INK_MUTED))
        p.drawText(QRectF(0, 154, 244, 16), Qt.AlignmentFlag.AlignLeft,
                   f"average per cell, largest {top:.2f} px")

    def set_mode(self, uncertainty: bool, residuals: bool = False):
        self.show_residuals = residuals
        if residuals:
            self.show_uncertainty = False
            self.setToolTip(
                "What is left over after the fit, averaged over each part of the image: where the points were "
                "found minus where the result puts them. The dot is the far end of each line.\n"
                "With a lens model that fits, the lines are short and point every which way. "
                "Red lines are 0.1 px and longer.")
            self.update()
            return
        self.show_uncertainty = uncertainty
        self.setToolTip(
            "How far a viewing direction may land from where the result puts it, one standard deviation.\n"
            "Light: under 0.05 px. Mid: under 0.2 px. Dark: under 0.5 px. Red: 0.5 px and more.\n"
            "From the scatter of the detected points; a slight turn of the whole camera is not counted."
            if uncertainty else "")
        self.update()

    def _paint_uncertainty(self, p: QPainter, area: QRectF):
        sd = self._spread
        if sd is None:
            p.setPen(QPen(QColor(T.HAIRLINE), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(area.adjusted(0.5, 0.5, -0.5, -0.5))
            p.setPen(QColor(T.INK_SOFT))
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "No result yet")
            return
        rows, cols = sd.shape
        cw, ch = area.width() / cols, area.height() / rows
        for r in range(rows):
            for c in range(cols):
                v = float(sd[r, c])
                if not np.isfinite(v):
                    continue  # outside what the lens model reaches
                shade = (T.HAIRLINE if v < 0.05 else T.LINE_STRONG if v < 0.2
                         else T.INK if v < 0.5 else T.RED_TEXT)
                p.setPen(QPen(QColor(shade), 1))
                p.setBrush(QColor(shade))
                p.drawRect(QRectF(c * cw + 1, r * ch + 1, cw - 2, ch - 2))
        p.setPen(QColor(T.INK_MUTED))
        p.drawText(QRectF(0, 154, 244, 16), Qt.AlignmentFlag.AlignLeft,
                   f"typical ±{np.nanmedian(sd):.2f} px · worst ±{np.nanmax(sd):.2f} px")

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setFont(T.mono(11))
        area = QRectF(0, 0, 240, 150)
        if self.show_residuals:
            self._paint_residuals(p, area)
            return
        if self.show_uncertainty:
            self._paint_uncertainty(p, area)
            return
        if self._grid is None:
            p.setPen(QPen(QColor(T.HAIRLINE), 1))
            p.drawRect(area.adjusted(0.5, 0.5, -0.5, -0.5))
            p.setPen(QColor(T.INK_SOFT))
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "Nothing detected yet")
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
                   f"{self._share * 100:.0f} % of the image is covered")


RESIDUAL_GAIN = 30  # residual vectors are drawn this many times longer than they are


class ImageView(QWidget):
    """Shows one image with its detected corners and, after calibration, the residuals."""

    corrected_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self._pix: QPixmap | None = None
        self._corners = np.zeros((0, 2))
        self._rings = np.zeros((0, 2))
        self._residuals: np.ndarray | None = None
        self._message = ""
        self._caption = ""
        self._allowed = False
        self.corrected_btn = button("Corrected view")
        self.corrected_btn.setParent(self)
        self.corrected_btn.setCheckable(True)
        self.corrected_btn.setToolTip("Show the image with the lens distortion of this result removed")
        self.corrected_btn.setFixedSize(150, 32)
        self.corrected_btn.hide()
        self.corrected_btn.toggled.connect(self.corrected_toggled)

    @property
    def corrected(self) -> bool:
        return self._allowed and self.corrected_btn.isChecked()

    def allow_corrected(self, allowed: bool) -> None:
        """The switch is offered only while there is a result to correct with."""
        self._allowed = allowed
        self.corrected_btn.setVisible(allowed)
        self._place_button()
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_button()

    def _place_button(self) -> None:
        # In the caption strip under the image, right-aligned; never on top of the image.
        b = self.corrected_btn
        b.move(self.width() - b.width() - 16, self.height() - 16 - 36)

    def set_message(self, text: str):
        self._pix, self._message, self._caption = None, text, ""
        self.update()

    def set_image(self, gray: np.ndarray | None, corners: np.ndarray, residuals: np.ndarray | None, caption: str):
        if gray is None:
            self.set_message("This image could not be read.\n\nIt may have been moved or renamed since it was added.")
            return
        h, w = gray.shape[:2]
        img = QImage(np.ascontiguousarray(gray).data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        self._pix = QPixmap.fromImage(img)
        self._corners, self._residuals, self._caption = corners, residuals, caption
        self._rings = np.zeros((0, 2))
        self.update()

    def set_rings(self, points: np.ndarray):
        """Points to circle in the current image, in image pixels: board points that look misplaced."""
        self._rings = np.asarray(points, np.float64).reshape(-1, 2)
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
        for cx, cy in self._rings:
            pt = QPointF(x0 + (cx + 0.5) * s, y0 + (cy + 0.5) * s)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#FFFFFF"), 4))
            p.drawEllipse(pt, 13, 13)
            p.setPen(QPen(QColor(T.RED), 2))
            p.drawEllipse(pt, 13, 13)
        p.setPen(QColor(T.INK_MUTED))
        p.setFont(T.mono(12))
        room = self.corrected_btn.width() + 16 if self._allowed else 0
        p.drawText(QRectF(pad, self.height() - pad - 34, self.width() - 2 * pad - room, 36),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, self._caption)
