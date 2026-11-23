"""Calibration report as a PDF (A4), drawn with the same visual identity as the app."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from PySide6.QtCore import QMarginsF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen
from PySide6.QtSvg import QSvgRenderer

from .. import VERSION_NAME
from ..core.calibrate import CalibrationResult, coverage_grid
from ..core.detect import Detection
from ..core.findings import CHECK, NOTE, PASS, error_outside_fit, findings, group_findings, model_sentence, rig_findings
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
    kind: str = "Single camera",
) -> int:
    """Write the report. Returns the number of pages."""
    writer = QPdfWriter(str(path))
    writer.setResolution(96)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)
    writer.setTitle(f"{project_name} calibration report, revision {revision}")
    writer.setCreator(f"Plumbline {VERSION_NAME} by Calib.Tools")

    rows = _image_rows(result, detections)
    pages = 2 + max(1, -(-len(rows) // ROWS_PER_PAGE))
    p = QPainter(writer)
    try:
        # Map the 794 x 1123 layout onto whatever device size the writer reports.
        p.scale(writer.width() / PAGE_W, writer.height() / PAGE_H)
        _summary_page(p, result, detections, project_name, revision, 1, pages, kind)
        writer.newPage()
        _maps_page(p, result, detections, project_name, 2, pages)
        for n in range(pages - 2):
            writer.newPage()
            _table_page(p, rows[n * ROWS_PER_PAGE : (n + 1) * ROWS_PER_PAGE], project_name, n + 3, pages)
    finally:
        p.end()
    return pages


def write_rig_report(rig, first: list[Detection], second: list[Detection], path: str | Path,
                     project_name: str = "Calibration", revision: int = 1,
                     camera_reports: tuple[str, str] | None = None, check=None) -> int:
    """Report on two cameras together: where camera 2 sits, and every pair. Returns the number of pages."""
    writer = QPdfWriter(str(path))
    writer.setResolution(96)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)
    writer.setTitle(f"{project_name} calibration report for two cameras, revision {revision}")
    writer.setCreator(f"Plumbline {VERSION_NAME} by Calib.Tools")
    used = {q.index: q for q in rig.pairs}
    rows = []
    for i in range(max(len(first), len(second))):
        names = [Path(d[i].path).name if i < len(d) else "–" for d in (first, second)]
        q = used.get(i)
        if q is not None:
            status = "used"
        elif i in rig.first.excluded or i in rig.second.excluded:
            status = "excluded"
        else:
            status = "no pair"
        rows.append((i + 1, names[0], names[1], q.points if q else None, q.rms_point if q else None, status))
    pages = 1 + max(1, -(-len(rows) // ROWS_PER_PAGE))
    p = QPainter(writer)
    try:
        p.scale(writer.width() / PAGE_W, writer.height() / PAGE_H)
        _rig_page(p, rig, rows, project_name, revision, 1, pages, camera_reports, check)
        for n in range(pages - 1):
            writer.newPage()
            _pairs_page(p, rows[n * ROWS_PER_PAGE: (n + 1) * ROWS_PER_PAGE], project_name, n + 2, pages)
    finally:
        p.end()
    return pages


def write_group_report(group, detections: list[list[Detection]], path: str | Path,
                       project_name: str = "Calibration", revision: int = 1,
                       camera_reports: list[str] | None = None, check=None) -> int:
    """Report on three or more cameras together: where each sits, and every set. Returns the number of pages."""
    writer = QPdfWriter(str(path))
    writer.setResolution(96)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)
    writer.setTitle(f"{project_name} calibration report for {len(group.cameras)} cameras, revision {revision}")
    writer.setCreator(f"Plumbline {VERSION_NAME} by Calib.Tools")
    used = {q.index: q for q in group.sets}
    rows = []
    for i in range(max(len(d) for d in detections)):
        q = used.get(i)
        seen = [c for c, d in enumerate(detections) if i < len(d) and d[i].usable]
        if q is not None:
            status = "used"
        elif any(i in cam.excluded for cam in group.cameras):
            status = "excluded"
        else:
            status = "alone"
        rows.append((i + 1, q.cameras if q else tuple(seen), q.points if q else None, q.rms_point if q else None, status))
    pages = 2 + max(1, -(-len(rows) // ROWS_PER_PAGE))
    p = QPainter(writer)
    try:
        p.scale(writer.width() / PAGE_W, writer.height() / PAGE_H)
        _group_page(p, group, rows, project_name, revision, 1, pages, camera_reports)
        writer.newPage()
        _group_detail_page(p, group, rows, project_name, 2, pages, check)
        for n in range(pages - 2):
            writer.newPage()
            _sets_page(p, rows[n * ROWS_PER_PAGE: (n + 1) * ROWS_PER_PAGE], project_name, n + 3, pages)
    finally:
        p.end()
    return pages


def _ruled_table(p, y, cols, rows, row_h=23.0) -> float:
    """A table over the page width: a tinted head, one ruled line per row. Returns the height used."""
    W = PAGE_W - 2 * M
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(T.RED_TINT))
    p.drawRect(QRectF(M, y, W, row_h))

    def cell(ci, text, yy, font, color):
        _, x0, share, align = cols[ci]
        x = M + x0 * W + (10 if align == Qt.AlignmentFlag.AlignLeft else 0)
        _text(p, x, yy + 6, share * W - 10, text, font, color, align=align)

    for ci, (title, _x, _w, _a) in enumerate(cols):
        cell(ci, title, y, T.mono(10, 500), T.RED_TEXT)
    top = y
    y += row_h
    for cells, color in rows:
        for ci, text in enumerate(cells):
            cell(ci, text, y, T.mono(11), color)
        y += row_h
        _line(p, M, y, M + W, y)
    return y - top


def _group_page(p, group, rows, project_name, revision, page, pages, camera_reports) -> None:
    _frame(p, page, pages, project_name)
    W = PAGE_W - 2 * M
    count = len(group.cameras)
    y = M + 48
    y += _text(p, M, y, W, project_name, T.mono(22, 600)) + 4
    y += _text(p, M, y, W, f"{count} cameras  ·  {date.today().isoformat()}  \u00b7  Revision {revision}",
               T.sans(12), T.INK_MUTED) + 22

    _eyebrow(p, M, y, "RMS reprojection error, all cameras")
    _eyebrow(p, M + W / 2 + 12, y, "Sets")
    y += 16
    h1 = _text(p, M, y, W / 2, f"{group.rms_point:.3f} px", T.mono(32, 600))
    points = sum(q.points for q in group.sets)
    _text(p, M + W / 2 + 12, y + 2, W / 2 - 12, f"{len(group.sets)} of {len(rows)} used  ·  {points:,} points",
          T.mono(13, 600))
    _text(p, M + W / 2 + 12, y + 22, W / 2 - 12,
          "A set is the images taken at the same moment, one per camera. It is used when at least two of them "
          "show the target.", T.sans(11), T.INK_MUTED, wrap=True)
    y += h1 + 2
    y += _text(p, M, y, W / 2 - 12, "Per point, over all images of every set that is used.",
               T.sans(11), T.INK_MUTED, wrap=True) + 26

    _eyebrow(p, M, y, "Where each camera sits, seen from camera 1")
    y += 16
    right = Qt.AlignmentFlag.AlignRight
    cols = [("Camera", 0.00, 0.12, Qt.AlignmentFlag.AlignLeft), ("Apart, mm", 0.12, 0.18, right),
            ("Turned, °", 0.30, 0.18, right), ("X (right), mm", 0.48, 0.17, right),
            ("Y (down), mm", 0.65, 0.17, right), ("Z (forward), mm", 0.82, 0.18, right)]

    def pm(value, sd, digits=2):
        if not sd == sd:
            return f"{value:.{digits}f}"
        return f"{value:.{digits}f} ± {sd:.{digits if round(sd, digits) > 0 else digits + 1}f}"

    body = []
    for q in group.placed:
        x, yy, z = q.position
        body.append(((f"{q.camera + 1}", pm(q.apart_mm, q.apart_std_mm), pm(q.turned_deg, q.turned_std_deg, 3),
                      pm(x, q.position_std[0]), pm(yy, q.position_std[1]), pm(z, q.position_std[2])), T.INK))
    y += _ruled_table(p, y, cols, body) + 8
    y += _text(p, M, y, W,
               "X, Y and Z are where a camera sits in camera 1's own directions: to its right, downwards, and the "
               "way it looks. ± is one standard deviation, from the scatter of the detected points and from how "
               "well each camera's own parameters are known. It does not cover a target that is not flat or not "
               "the size entered, or cameras that were not triggered together.",
               T.sans(11), T.INK_MUTED, wrap=True) + 20

    _eyebrow(p, M, y, "The cameras")
    y += 16
    for n, cam in enumerate(group.cameras, 1):
        _text(p, M, y, 70, f"Camera {n}", T.sans(11, 600))
        line = (f"{cam.image_width} × {cam.image_height} px  ·  fx {cam.fx:.2f}  fy {cam.fy:.2f}  cx {cam.cx:.2f}  "
                f"cy {cam.cy:.2f}  ·  {cam.rms_point:.3f} px alone")
        y += _text(p, M + 76, y, W - 76, line, T.mono(11)) + 2
        more = model_sentence(cam) + (f"  Its own report: {camera_reports[n - 1]}" if camera_reports else "")
        y += _text(p, M + 76, y, W - 76, more, T.sans(10), T.INK_MUTED, wrap=True) + 6


def _check_block(p, y, check, name: str, who: str, what: str) -> float:
    """The check with images that took no part. Returns where the page goes on."""
    W = PAGE_W - 2 * M
    _eyebrow(p, M, y, "Checked with images that took no part", W)
    y += 16
    off = (check.scale - 1) * 100
    y += _kv_table(p, M, y, W, [
        ("Lengths come out", f"{off:+.3f} %" if abs(off) >= 0.0005 else "0.000 %"),
        (name, f"{len(check.pairs)} used" + (f", {len(check.skipped)} not" if check.skipped else "")),
        ("Length error, RMS", f"{check.length_error_rms_mm:.2f} mm"),
        ("Largest error", f"{check.length_error_max_mm:.2f} mm"),
        ("Out of flat, RMS", f"{check.flatness_rms_mm:.2f} mm"),
        ("Lengths compared", f"{check.spans:,}, up to {check.longest_mm:.0f} mm")], cols=2) + 6
    return y + _text(p, M, y, W,
                     f"{who} measured lengths between points of the target in {what} of images that were not "
                     "used for the calibration; these are compared with the target's own dimensions. + means measured "
                     "too long. Out of flat is how far the measured points are from one plane.",
                     T.sans(11), T.INK_MUTED, wrap=True) + 14


def _group_detail_page(p, group, rows, project_name, page, pages, check=None) -> None:
    _frame(p, page, pages, project_name)
    W = PAGE_W - 2 * M
    y = M + 48
    _eyebrow(p, M, y, "Error per set")
    y += 16
    _error_chart(p, QRectF(M, y, W, 130), [(r[3], r[4] == "excluded") for r in rows])
    y += 130 + 4
    y += _text(p, M, y, W, "px per set, RMS per point over all its images, in set order. A set without a bar "
               "takes no part.", T.sans(10), T.INK_MUTED) + 20
    if check is not None:
        y = _check_block(p, y, check, "Sets", "The cameras", "sets") + 6

    _eyebrow(p, M, y, "Findings")
    y += 16
    for tag, text in group_findings(group):
        _text(p, M, y + 1, 50, tag, T.mono(10, 600), TAG_COLOR[tag])
        y += _text(p, M + 56, y, W - 56, text, T.sans(11), wrap=True) + 7
    y += 16

    _eyebrow(p, M, y, "As in the files")
    y += 14
    y += _text(p, M, y, W, "A point x1 in camera 1 coordinates is Rn · x1 + Tn in camera n coordinates.",
               T.sans(10), T.INK_MUTED) + 6
    for q in group.placed:
        n = q.camera + 1
        y += _kv_table(p, M, y, W, [(f"R{n}, row {i + 1}", "  ".join(f"{v: .6f}" for v in q.rotation[i])) for i in range(3)]
                       + [(f"T{n}, mm", "  ".join(f"{v: .3f}" for v in q.translation))], cols=2, row_h=20.0) + 8


def _sets_page(p, rows, project_name, page, pages) -> None:
    _frame(p, page, pages, project_name)
    y = M + 48
    _eyebrow(p, M, y, "Sets")
    y += 18
    right = Qt.AlignmentFlag.AlignRight
    cols = [("Set", 0.00, 0.10, Qt.AlignmentFlag.AlignLeft), ("Cameras with the target", 0.10, 0.40, Qt.AlignmentFlag.AlignLeft),
            ("Points", 0.50, 0.14, right), ("RMS, px", 0.64, 0.14, right), ("Status", 0.84, 0.16, Qt.AlignmentFlag.AlignLeft)]
    body = []
    for number, cams, points, rms, status in rows:
        body.append(((f"{number:02d}", "  ".join(str(c + 1) for c in cams) or "–", str(points) if points is not None else "–",
                      f"{rms:.3f}" if rms is not None else "–", status), T.INK if status == "used" else T.RED_TEXT))
    _ruled_table(p, y, cols, body)


def _rig_page(p, rig, rows, project_name, revision, page, pages, camera_reports, check=None) -> None:
    _frame(p, page, pages, project_name)
    W = PAGE_W - 2 * M
    y = M + 48
    y += _text(p, M, y, W, project_name, T.mono(22, 600)) + 4
    y += _text(p, M, y, W, f"Two cameras  ·  {date.today().isoformat()}  \u00b7  Revision {revision}",
               T.sans(12), T.INK_MUTED) + 22

    _eyebrow(p, M, y, "RMS reprojection error, both cameras")
    _eyebrow(p, M + W / 2 + 12, y, "Pairs")
    y += 16
    h1 = _text(p, M, y, W / 2, f"{rig.rms_point:.3f} px", T.mono(32, 600))
    points = sum(q.points for q in rig.pairs)
    _text(p, M + W / 2 + 12, y + 2, W / 2 - 12, f"{len(rig.pairs)} of {len(rows)} used  ·  {points:,} points",
          T.mono(13, 600))
    _text(p, M + W / 2 + 12, y + 22, W / 2 - 12,
          "A pair is the two images taken at the same moment. Points are counted in both images.",
          T.sans(11), T.INK_MUTED, wrap=True)
    y += h1 + 2
    y += _text(p, M, y, W / 2 - 12,
               f"Per point, over both images of every pair. Camera 1 alone {rig.first.rms_point:.3f} px, "
               f"camera 2 alone {rig.second.rms_point:.3f} px.", T.sans(11), T.INK_MUTED, wrap=True) + 20

    # Where camera 2 sits.
    _eyebrow(p, M, y, "Camera 2, seen from camera 1")
    _eyebrow(p, M + (W - 16) / 2 + 16, y, "As in the files: x2 = R · x1 + T")
    y += 16
    half = (W - 16) / 2
    x, yy, z = rig.position
    known = rig.apart_std_mm == rig.apart_std_mm
    h_a = _kv_table(p, M, y, half, [
        ("Apart, mm", f"{rig.apart_mm:.2f} ± {rig.apart_std_mm:.2f}" if known else f"{rig.apart_mm:.2f}"),
        ("Turned, degrees", f"{rig.turned_deg:.3f} ± {rig.turned_std_deg:.3f}" if known else f"{rig.turned_deg:.3f}"),
        *[(name, f"{v:.2f} ± {sd:.{2 if round(sd, 2) > 0 else 3}f}" if known and sd == sd else f"{v:.2f}")
          for name, v, sd in zip(("X (right), mm", "Y (down), mm", "Z (forward), mm"), (x, yy, z), rig.position_std)]],
        cols=1)
    R, Tv = rig.rotation, rig.translation
    h_b = _kv_table(p, M + half + 16, y, half,
                    [(f"R, row {i + 1}", "  ".join(f"{v: .6f}" for v in R[i])) for i in range(3)]
                    + [("T, mm", "  ".join(f"{v: .3f}" for v in Tv))], cols=1)
    y += max(h_a, h_b) + 6
    y += _text(p, M, y, W,
               "X, Y and Z are where camera 2 sits in camera 1's own directions: to its right, downwards, and the "
               "way it looks. ± is one standard deviation, from the scatter of the detected points and from how "
               "well each camera's own parameters are known. It does not cover a target that is not flat or not "
               "the size entered, or cameras that were not triggered together.",
               T.sans(11), T.INK_MUTED, wrap=True) + 14

    # The two cameras side by side.
    _eyebrow(p, M, y, "The cameras")
    y += 16
    for n, cam in ((1, rig.first), (2, rig.second)):
        _text(p, M, y, 70, f"Camera {n}", T.sans(11, 600))
        line = (f"{cam.image_width} × {cam.image_height} px  ·  fx {cam.fx:.2f}  fy {cam.fy:.2f}  cx {cam.cx:.2f}  "
                f"cy {cam.cy:.2f}  ·  {cam.rms_point:.3f} px alone")
        h = _text(p, M + 76, y, W - 76, line, T.mono(11))
        y += h + 2
        more = model_sentence(cam) + (f"  Its own report: {camera_reports[n - 1]}" if camera_reports else "")
        y += _text(p, M + 76, y, W - 76, more, T.sans(10), T.INK_MUTED, wrap=True) + 6
    y += 12

    if check is not None:
        y = _check_block(p, y, check, "Pairs", "The two cameras", "pairs")

    chart_h = 130 if check is None else 96
    _eyebrow(p, M, y, "Error per pair")
    y += 16
    _error_chart(p, QRectF(M, y, W, chart_h), [(r[4], r[5] == "excluded") for r in rows])
    y += chart_h + 4
    y += _text(p, M, y, W, "px per pair, RMS per point over both images, in pair order. A pair without a bar "
               "takes no part.", T.sans(10), T.INK_MUTED) + 18

    _eyebrow(p, M, y, "Findings")
    y += 16
    for tag, text in rig_findings(rig):
        _text(p, M, y + 1, 50, tag, T.mono(10, 600), TAG_COLOR[tag])
        y += _text(p, M + 56, y, W - 56, text, T.sans(11), wrap=True) + 7


def _pairs_page(p, rows, project_name, page, pages) -> None:
    _frame(p, page, pages, project_name)
    W = PAGE_W - 2 * M
    y = M + 48
    _eyebrow(p, M, y, "Pairs")
    y += 18
    cols = [("Pair", 0.00, 0.08, Qt.AlignmentFlag.AlignLeft), ("Camera 1", 0.08, 0.28, Qt.AlignmentFlag.AlignLeft),
            ("Camera 2", 0.36, 0.28, Qt.AlignmentFlag.AlignLeft), ("Points", 0.64, 0.10, Qt.AlignmentFlag.AlignRight),
            ("RMS, px", 0.74, 0.12, Qt.AlignmentFlag.AlignRight), ("Status", 0.89, 0.11, Qt.AlignmentFlag.AlignLeft)]
    row_h = 23.0
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(T.RED_TINT))
    p.drawRect(QRectF(M, y, W, row_h))

    def cell(ci, text, yy, font, color):
        _, x0, share, align = cols[ci]
        x = M + x0 * W + (10 if align == Qt.AlignmentFlag.AlignLeft else 0)
        _text(p, x, yy + 6, share * W - 10, text, font, color, align=align)

    def short(name: str) -> str:
        return name if len(name) <= 26 else name[:12] + "…" + name[-12:]

    for ci, (title, _x, _w, _a) in enumerate(cols):
        cell(ci, title, y, T.mono(10, 500), T.RED_TEXT)
    y += row_h
    for number, one, two, points, rms, status in rows:
        color = T.INK if status == "used" else T.RED_TEXT
        cell(0, f"{number:02d}", y, T.mono(11), color)
        cell(1, short(one), y, T.mono(11), color)
        cell(2, short(two), y, T.mono(11), color)
        cell(3, str(points) if points is not None else "–", y, T.mono(11), color)
        cell(4, f"{rms:.3f}" if rms is not None else "–", y, T.mono(11), color)
        cell(5, status, y, T.sans(11), color)
        y += row_h
        _line(p, M, y, M + W, y)


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
def _summary_page(p, r: CalibrationResult, detections, project_name, revision, page, pages,
                  kind: str = "Single camera") -> None:
    _frame(p, page, pages, project_name)
    W = PAGE_W - 2 * M
    y = M + 48

    y += _text(p, M, y, W, project_name, T.mono(22, 600)) + 4
    y += _text(p, M, y, W, f"{kind}  ·  {r.image_width} × {r.image_height} px  ·  "
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
    u = r.uncertainty

    def pm(name: str, value: float, digits: int) -> str:
        sd = u.std_of(name) if u is not None else None
        return f"{value:.{digits}f}" if sd is None else f"{value:.{digits}f} ± {sd:.{digits}f}"

    h_a = _kv_table(p, M, y, half, [("fx, px", pm("fx", r.fx, 3)), ("fy, px", pm("fy", r.fy, 3)),
                                    ("cx, px", pm("cx", r.cx, 3)), ("cy, px", pm("cy", r.cy, 3))],
                    cols=1 if u is not None else 2)
    h_b = _kv_table(p, M + half + 16, y, half,
                    [(k, pm(k, float(v), 6)) for k, v in zip(r.distortion_names, r.distortion)],
                    cols=1 if u is not None else 2)
    y += max(h_a, h_b) + 6
    if u is not None:
        y += _text(p, M, y, W,
                   "± is one standard deviation, worked out from the scatter of the detected points "
                   f"({u.noise_px:.3f} px per coordinate). It does not cover a board that is not flat or "
                   "not the size entered, or a lens model that does not fit the lens.",
                   T.sans(11), T.INK_MUTED, wrap=True)
    y += 14

    # Target and method.
    _eyebrow(p, M, y, "Target and method")
    y += 16
    facts = [
        ("Target", r.target.describe()),
        ("Lens model", model_sentence(r)),
        ("Software", f"Plumbline {VERSION_NAME}, OpenCV {r.opencv_version}"),
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

    # Findings, full width. The maps they refer to are on the next page.
    _eyebrow(p, M, y, "Findings")
    y += 16
    active = [d for i, d in enumerate(detections) if i in by_index]
    _grid, share = coverage_grid(active, r.image_width, r.image_height)
    for tag, text in findings(r, share):
        _text(p, M, y + 1, 50, tag, T.mono(10, 600), TAG_COLOR[tag])
        y += _text(p, M + 56, y, W - 56, text, T.sans(11), wrap=True) + 7


def _maps_page(p, r: CalibrationResult, detections, project_name, page, pages) -> None:
    """Three maps of the image: where the points fell, how sure the result is there, what is left over."""
    _frame(p, page, pages, project_name)
    W = PAGE_W - 2 * M
    y = M + 48
    y += _text(p, M, y, W, "Across the image", T.mono(18, 600)) + 4
    y += _text(p, M, y, W, f"Each map shows the whole image, {r.image_width} × {r.image_height} px, "
               "divided into cells.", T.sans(12), T.INK_MUTED) + 22

    gap = 24.0
    mw = (W - gap) / 2
    mh = min(mw * r.image_height / max(r.image_width, 1), 300.0)
    mw_used = mh * r.image_width / max(r.image_height, 1)
    x2 = M + mw + gap
    used = {v.index for v in r.views}
    grid, share = coverage_grid([d for i, d in enumerate(detections) if i in used], r.image_width, r.image_height)

    # Row 1: coverage and uncertainty.
    _eyebrow(p, M, y, "Coverage")
    _eyebrow(p, x2, y, "Uncertainty")
    y += 16
    _coverage(p, QRectF(M, y, mw_used, mh), grid)
    sd = r.projection.sd if r.projection is not None else None
    _uncertainty_map(p, QRectF(x2, y, mw_used, mh), sd)
    y += mh + 8
    h1 = _text(p, M, y, mw, f"{share * 100:.0f} % of the image is covered by {r.target.point_word}. "
               "Dark: many. Light: few. Red outline: none; the result is not checked there.",
               T.sans(10), T.INK_MUTED, wrap=True)
    if sd is None:
        note = "Not worked out for this result."
    else:
        note = (f"Typical ±{np.nanmedian(sd):.2f} px, worst ±{np.nanmax(sd):.2f} px. How far a viewing direction "
                "may land from where the result puts it, one standard deviation. Light: under 0.05 px. "
                "Mid: under 0.2 px. Dark: under 0.5 px. Red: 0.5 px and more. From the scatter of the detected "
                "points; a slight turn of the whole camera is not counted.")
    h2 = _text(p, x2, y, mw, note, T.sans(10), T.INK_MUTED, wrap=True)
    y += max(h1, h2) + 26

    # Row 2: residuals, and how to read the page.
    _eyebrow(p, M, y, "Residuals")
    _eyebrow(p, x2, y, "How to read this page")
    y += 16
    lean = r.pattern.mean if r.pattern is not None else None
    _residual_map(p, QRectF(M, y, mw_used, mh), lean)
    ty = y
    for head, body in (
        ("Coverage", "Parts of the image without points are fitted by the lens model alone. "
                     "Add views that put the target there."),
        ("Uncertainty", "Where this map is dark or red, the same camera calibrated again from new images "
                        "would give a visibly different answer. More views there bring it down."),
        ("Residuals", "With a lens model that fits, the lines are short and point every which way. "
                      "Lines that agree with their neighbours mean the model cannot follow the lens, "
                      "or the board is not the shape entered."),
    ):
        ty += _text(p, x2, ty, mw, head, T.sans(11, 600)) + 2
        ty += _text(p, x2, ty, mw, body, T.sans(11), wrap=True) + 10
    y += mh + 8
    if lean is None:
        note = "Not worked out for this result."
    else:
        size = np.linalg.norm(lean, axis=2)
        top = float(np.nanmax(size)) if np.isfinite(size).any() else 0.0
        note = (f"Largest {top:.2f} px. What is left over after the fit, averaged over each cell: where the "
                "points were found minus where the result puts them. The dot is the far end of each line; "
                "lines are drawn enlarged. Red: 0.1 px and more.")
        if r.pattern.has_pattern:
            note += f" The pattern is {r.pattern.strength:.1f} times what noise alone would leave."
    _text(p, M, y, mw, note, T.sans(10), T.INK_MUTED, wrap=True)


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


def _map_border(p, area: QRectF, empty: str | None = None) -> None:
    p.setPen(QPen(QColor(T.HAIRLINE), 0.8))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(area)
    if empty:
        _text(p, area.left(), area.center().y() - 8, area.width(), empty, T.sans(11), T.INK_MUTED,
              Qt.AlignmentFlag.AlignHCenter)


def _uncertainty_map(p, area: QRectF, sd: np.ndarray | None) -> None:
    if sd is None:
        _map_border(p, area, "No uncertainty for this result")
        return
    _map_border(p, area)   # cells the lens model does not reach stay empty; the frame shows where the image ends
    rows, cols = sd.shape
    cw, ch = area.width() / cols, area.height() / rows
    for r in range(rows):
        for c in range(cols):
            v = float(sd[r, c])
            if not np.isfinite(v):
                continue  # outside what the lens model reaches
            shade = T.HAIRLINE if v < 0.05 else T.LINE_STRONG if v < 0.2 else T.INK if v < 0.5 else T.RED_TEXT
            p.setPen(QPen(QColor(shade), 0.8))
            p.setBrush(QColor(shade))
            p.drawRect(QRectF(area.left() + c * cw + 1, area.top() + r * ch + 1, cw - 2, ch - 2))


def _residual_map(p, area: QRectF, lean: np.ndarray | None) -> None:
    if lean is None:
        _map_border(p, area, "No residual map for this result")
        return
    _map_border(p, area)
    rows, cols = lean.shape[:2]
    cw, ch = area.width() / cols, area.height() / rows
    size = np.linalg.norm(lean, axis=2)
    finite = size[np.isfinite(size)]
    scale = float(np.percentile(finite, 85)) if finite.size else 0.0
    reach = 1.6 * min(cw, ch)                      # same drawing rule as the map in the window
    gain = min(cw, ch) / max(scale, 0.05)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for r in range(rows):
        for c in range(cols):
            if not np.isfinite(size[r, c]):
                continue
            mid = QPointF(area.left() + (c + 0.5) * cw, area.top() + (r + 0.5) * ch)
            dx, dy = (float(v) for v in lean[r, c] * gain)
            length = float(np.hypot(dx, dy))
            if length > reach:
                dx, dy = dx * reach / length, dy * reach / length
            strong = size[r, c] >= 0.1
            colour = QColor(T.RED if strong else T.INK)
            p.setPen(QPen(colour, 1.4 if strong else 1.0))
            p.setBrush(colour)
            p.drawLine(QPointF(mid.x() - dx / 2, mid.y() - dy / 2), QPointF(mid.x() + dx / 2, mid.y() + dy / 2))
            p.drawEllipse(QPointF(mid.x() + dx / 2, mid.y() + dy / 2), 1.6, 1.6)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
