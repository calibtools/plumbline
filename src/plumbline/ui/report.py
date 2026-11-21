"""Calibration report as a PDF (A4), drawn with the same visual identity as the app."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from PySide6.QtCore import QMarginsF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen
from PySide6.QtSvg import QSvgRenderer

from .. import __version__
from ..core.calibrate import CalibrationResult, coverage_grid
from ..core.detect import Detection
from ..core.findings import CHECK, NOTE, PASS, error_outside_fit, findings, model_sentence
from . import theme as T
from .widgets import _nice_ticks

PAGE_W, PAGE_H = 794.0, 1123.0  # A4 at 96 px per inch
M = 48.0  # page margin
ROWS_PER_PAGE = 38
TAG_COLOR = {PASS: T.GREEN_TEXT, CHECK: T.RED_TEXT, NOTE: T.INK_MUTED}


def write_report(
    result: CalibrationResult,
    detections: list[Detection],
    path: str | Path,
    project_name: str = "Calibration",
    revision: int = 1,
) -> int:
    """Write the report. Returns the number of pages."""
    writer = QPdfWriter(str(path))
    writer.setResolution(96)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)
    writer.setTitle(f"{project_name} calibration report, revision {revision}")
    writer.setCreator(f"Plumbline {__version__} by Calib.Tools")

    rows = _image_rows(result, detections)
    pages = 1 + max(1, -(-len(rows) // ROWS_PER_PAGE))
    p = QPainter(writer)
    try:
        # Map the 794 x 1123 layout onto whatever device size the writer reports.
        p.scale(writer.width() / PAGE_W, writer.height() / PAGE_H)
        _summary_page(p, result, detections, project_name, revision, 1, pages)
        for n in range(pages - 1):
            writer.newPage()
            _table_page(p, rows[n * ROWS_PER_PAGE : (n + 1) * ROWS_PER_PAGE], project_name, n + 2, pages)
    finally:
        p.end()
    return pages


# ------------------------------------------------------------------ helpers
def _text(p, x, y, w, text, font, color=T.INK, align=Qt.AlignmentFlag.AlignLeft, wrap=False) -> float:
    """Draw text with its top at y. Returns the height used."""
    p.setFont(font)
    p.setPen(QColor(color))
    flags = align | Qt.AlignmentFlag.AlignTop | (Qt.TextFlag.TextWordWrap if wrap else Qt.TextFlag.TextSingleLine)
    fm = QFontMetricsF(font, p.device())
    box = fm.boundingRect(QRectF(0, 0, w, 10000), flags, text)
    p.drawText(QRectF(x, y, w, box.height() + 2), flags | Qt.TextFlag.TextDontClip, text)
    return box.height()


def _line(p, x1, y1, x2, y2, color=T.HAIRLINE, width=1.0):
    p.setPen(QPen(QColor(color), width))
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


def _eyebrow(p, x, y, text, w=300.0) -> float:
    return _text(p, x, y, w, text.upper(), T.mono(10, 500), T.INK_MUTED)


def _frame(p, page: int, pages: int, project_name: str) -> None:
    mark = QSvgRenderer(T.asset("wordmark.svg"))
    size = mark.defaultSize()
    h = 26.0
    mark.render(p, QRectF(M, M - 6, h * size.width() / size.height(), h))
    _text(p, PAGE_W - M - 300, M + 2, 300, "CALIBRATION REPORT", T.mono(11, 600), align=Qt.AlignmentFlag.AlignRight)
    _line(p, M, M + 30, PAGE_W - M, M + 30, T.INK)
    y = PAGE_H - M - 4
    _line(p, M, y - 10, PAGE_W - M, y - 10, T.HAIRLINE)
    _text(p, M, y - 4, 380, "Calibrated with Plumbline by Calib.Tools, maker of calibration targets.  calib.tools",
          T.sans(10), T.INK_MUTED)
    _text(p, PAGE_W - M - 200, y - 4, 200, f"{project_name}  ·  page {page} of {pages}",
          T.mono(10), T.INK_MUTED, align=Qt.AlignmentFlag.AlignRight)


def _kv_table(p, x, y, w, pairs, cols=2, row_h=22.0) -> float:
    """Key / value cells in a ruled box. Returns the height used."""
    rows = -(-len(pairs) // cols)
    h = rows * row_h
    p.setPen(QPen(QColor(T.HAIRLINE), 1))
    p.setBrush(QColor(T.SHEET))
    p.drawRect(QRectF(x, y, w, h))
    cw = w / cols
    for i, (k, v) in enumerate(pairs):
        r, c = divmod(i, cols)
        cx, cy = x + c * cw, y + r * row_h
        if r:
            _line(p, x, cy, x + w, cy)
        if c:
            _line(p, cx, y, cx, y + h)
        _text(p, cx + 10, cy + 5, cw / 2, k, T.mono(11), T.INK_MUTED)
        _text(p, cx + 10, cy + 5, cw - 20, v, T.mono(11, 600), align=Qt.AlignmentFlag.AlignRight)
    return h


def _image_rows(result: CalibrationResult, detections: list[Detection]):
    by_index = {v.index: v for v in result.views}
    rows = []
    for i, det in enumerate(detections):
        name = Path(det.path).name
        view = by_index.get(i)
        if view is not None:
            rows.append((name, det.count, view.rms_point, view.max_error, "used"))
        elif i in result.excluded:
            rows.append((name, det.count, error_outside_fit(result, det), None, "excluded"))
        else:
            rows.append((name, det.count, None, None, "no usable detection"))
    return rows


# -------------------------------------------------------------------- pages
def _summary_page(p, r: CalibrationResult, detections, project_name, revision, page, pages) -> None:
    _frame(p, page, pages, project_name)
    W = PAGE_W - 2 * M
    y = M + 48

    y += _text(p, M, y, W, project_name, T.mono(22, 600)) + 4
    y += _text(p, M, y, W, f"Single camera  ·  {r.image_width} × {r.image_height} px  ·  "
               f"{date.today().isoformat()}  \u00b7  Revision {revision}", T.sans(12), T.INK_MUTED) + 22

    # RMS block, with what went into it on the right.
    _eyebrow(p, M, y, "RMS reprojection error")
    _eyebrow(p, M + W / 2 + 12, y, "Images")
    y += 16
    h1 = _text(p, M, y, W / 2, f"{r.rms_point:.3f} px", T.mono(32, 600))
    _text(p, M + W / 2 + 12, y + 2, W / 2 - 12,
          f"{len(r.views)} of {len(detections)} used  ·  {r.corner_total:,} {r.target.point_word}", T.mono(13, 600))
    _text(p, M + W / 2 + 12, y + 22, W / 2 - 12,
          f"{len(r.excluded)} excluded by the user  ·  {len(r.skipped)} without a usable detection",
          T.sans(11), T.INK_MUTED, wrap=True)
    y += h1 + 2
    y += _text(p, M, y, W / 2 - 12,
               "Per point, sqrt(mean(dx² + dy²)), as OpenCV reports it. Counted per coordinate "
               f"(as mrcal does) the same result reads {r.rms_coordinate:.3f} px.",
               T.sans(11), T.INK_MUTED, wrap=True) + 20

    # Parameters.
    _eyebrow(p, M, y, "Parameters")
    y += 16
    half = (W - 16) / 2
    h_a = _kv_table(p, M, y, half, [("fx, px", f"{r.fx:.3f}"), ("fy, px", f"{r.fy:.3f}"),
                                    ("cx, px", f"{r.cx:.3f}"), ("cy, px", f"{r.cy:.3f}")])
    h_b = _kv_table(p, M + half + 16, y, half,
                    [(k, f"{v:.6f}") for k, v in zip(r.distortion_names, r.distortion)])
    y += max(h_a, h_b) + 20

    # Target and method.
    _eyebrow(p, M, y, "Target and method")
    y += 16
    facts = [
        ("Target", r.target.describe()),
        ("Lens model", model_sentence(r)),
        ("Software", f"Plumbline {__version__}, OpenCV {r.opencv_version}"),
    ]
    for k, v in facts:
        _text(p, M, y, 90, k, T.sans(11), T.INK_MUTED)
        y += _text(p, M + 96, y, W - 96, v, T.sans(11), wrap=True) + 5
    y += 14

    # Error per image.
    _eyebrow(p, M, y, "Error per image")
    y += 16
    by_index = {v.index: v for v in r.views}
    bars = []
    for i, det in enumerate(detections):
        if i in by_index:
            bars.append((by_index[i].rms_point, False))
        elif i in r.excluded:
            bars.append((error_outside_fit(r, det), True))
        else:
            bars.append((None, False))
    _error_chart(p, QRectF(M, y, W, 150), bars)
    y += 150 + 4
    y += _text(p, M, y, W, "px per image, RMS per point, in file order. Hollow bars are excluded images; "
               "a number above a bar means it runs off the scale.", T.sans(10), T.INK_MUTED) + 20

    # Coverage and findings side by side.
    _eyebrow(p, M, y, "Coverage")
    fx0 = M + 300
    _eyebrow(p, fx0, y, "Findings")
    y += 16
    active = [d for i, d in enumerate(detections) if i in by_index]
    grid, share = coverage_grid(active, r.image_width, r.image_height)
    _coverage(p, QRectF(M, y, 256, 160), grid)
    _text(p, M, y + 166, 280, f"{share * 100:.0f} % of the image is covered. Dark: many. Light: few. "
          "Red outline: none.", T.sans(10), T.INK_MUTED, wrap=True)
    fy = y
    for tag, text in findings(r, share):
        _text(p, fx0, fy + 1, 50, tag, T.mono(10, 600), TAG_COLOR[tag])
        fy += _text(p, fx0 + 52, fy, PAGE_W - M - fx0 - 52, text, T.sans(11), wrap=True) + 8


def _table_page(p, rows, project_name, page, pages) -> None:
    _frame(p, page, pages, project_name)
    W = PAGE_W - 2 * M
    y = M + 48
    _eyebrow(p, M, y, "Images")
    y += 18
    # title, left edge and width as shares of the table width, alignment
    cols = [("File", 0.00, 0.40, Qt.AlignmentFlag.AlignLeft), ("Points", 0.40, 0.12, Qt.AlignmentFlag.AlignRight),
            ("RMS, px", 0.52, 0.14, Qt.AlignmentFlag.AlignRight), ("Max, px", 0.66, 0.14, Qt.AlignmentFlag.AlignRight),
            ("Status", 0.84, 0.16, Qt.AlignmentFlag.AlignLeft)]
    row_h = 23.0
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(T.RED_TINT))
    p.drawRect(QRectF(M, y, W, row_h))
    def cell(ci, text, yy, font, color):
        _, x0, share, align = cols[ci]
        x = M + x0 * W + (10 if align == Qt.AlignmentFlag.AlignLeft else 0)
        _text(p, x, yy + 6, share * W - 10, text, font, color, align=align)
    for ci, (title, _x, _w, _a) in enumerate(cols):
        cell(ci, title, y, T.mono(10, 500), T.RED_TEXT)
    y += row_h
    for name, corners, rms, worst, status in rows:
        color = T.INK if status == "used" else T.RED_TEXT
        if len(name) > 38:
            name = name[:18] + "…" + name[-18:]
        cell(0, name, y, T.mono(11), color)
        cell(1, str(corners), y, T.mono(11), color)
        cell(2, f"{rms:.3f}" if rms is not None else "–", y, T.mono(11), color)
        cell(3, f"{worst:.3f}" if worst is not None else "–", y, T.mono(11), color)
        cell(4, status, y, T.sans(11), color)
        y += row_h
        _line(p, M, y, M + W, y)
    if any(s == "excluded" for *_rest, s in rows):
        _text(p, M, y + 8, W, "For excluded images the RMS is what the image would show under this result; "
              "it took no part in the solution.", T.sans(10), T.INK_MUTED, wrap=True)


def _error_chart(p, area: QRectF, bars) -> None:
    left, bottom_pad, top_pad = 34.0, 4.0, 12.0
    plot = QRectF(area.left() + left, area.top() + top_pad, area.width() - left, area.height() - top_pad - bottom_pad)
    used = [e for e, ex in bars if e is not None and not ex]
    known = [e for e, _ex in bars if e is not None]
    if not known:
        return
    top_val = max(0.1, 1.25 * max(used or known))
    for t in _nice_ticks(top_val):
        yy = plot.bottom() - plot.height() * t / top_val
        _line(p, plot.left(), yy, plot.right(), yy)
        _text(p, area.left(), yy - 7, left - 6, f"{t:g}", T.mono(10), T.INK_MUTED, align=Qt.AlignmentFlag.AlignRight)
    slot = plot.width() / len(bars)
    bw = max(2.0, min(18.0, slot * 0.62))
    for k, (err, excluded) in enumerate(bars):
        if err is None:
            continue
        x = plot.left() + slot * k + (slot - bw) / 2
        bh = plot.height() * min(err, top_val) / top_val
        if excluded:
            p.setPen(QPen(QColor(T.RED_TEXT), 1.2))
            p.setBrush(QColor(T.SHEET))
        else:
            p.setPen(QPen(QColor(T.INK), 0.5))
            p.setBrush(QColor(T.INK))
        p.drawRect(QRectF(x, plot.bottom() - bh, bw, bh))
        if err > top_val:
            _text(p, x - 20, plot.top() - 12, bw + 40, f"{err:.2f}", T.mono(9), T.RED_TEXT,
                  align=Qt.AlignmentFlag.AlignHCenter)
    _line(p, plot.left(), plot.bottom(), plot.right(), plot.bottom(), T.INK)


def _coverage(p, area: QRectF, grid: np.ndarray) -> None:
    rows, cols = grid.shape
    cw, ch = area.width() / cols, area.height() / rows
    positive = grid[grid > 0]
    hi = float(np.percentile(positive, 75)) if positive.size else 1.0
    for r in range(rows):
        for c in range(cols):
            v = int(grid[r, c])
            cell = QRectF(area.left() + c * cw + 1, area.top() + r * ch + 1, cw - 2, ch - 2)
            if v == 0:
                p.setPen(QPen(QColor(T.RED_TEXT), 0.8))
                p.setBrush(QColor(T.SHEET))
            else:
                shade = T.INK if v >= hi else T.LINE_STRONG if v >= hi / 3 else T.HAIRLINE
                p.setPen(QPen(QColor(shade), 0.8))
                p.setBrush(QColor(shade))
            p.drawRect(cell)
