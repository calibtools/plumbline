"""Excel workbook of a calibration: summary, one row per image, one row per corner."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .. import __version__
from .calibrate import CalibrationResult
from .detect import Detection
from .findings import error_outside_fit, model_sentence

HEAD_FILL = PatternFill("solid", start_color="FBE9EA")
HEAD_FONT = Font(bold=True, color="B3000E")
KEY_FONT = Font(color="4A4A47")
BOLD = Font(bold=True)
PX = "0.000"


def _header(ws, titles, widths):
    ws.append(titles)
    for c, width in enumerate(widths, start=1):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font = HEAD_FILL, HEAD_FONT
        cell.alignment = Alignment(horizontal="left" if c == 1 else "right")
        ws.column_dimensions[get_column_letter(c)].width = width
    ws.freeze_panes = "A2"


def write_xlsx(
    result: CalibrationResult,
    detections: list[Detection],
    path: str | Path,
    project_name: str = "Calibration",
    revision: int = 1,
) -> None:
    r = result
    wb = Workbook()

    # ---- Summary: one key per row, values as numbers so they can be used in formulas.
    ws = wb.active
    ws.title = "Summary"
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 58
    rows = [
        ("Project", project_name, None),
        ("Revision", revision, "0"),
        ("Date", date.today().isoformat(), None),
        ("Image width, px", r.image_width, "0"),
        ("Image height, px", r.image_height, "0"),
        ("Target", r.target.describe(), None),
        ("Lens model", model_sentence(r), None),
        ("RMS per point, px", r.rms_point, "0.0000"),
        ("RMS per coordinate, px", r.rms_coordinate, "0.0000"),
        ("Images used", len(r.views), "0"),
        ("Images excluded by the user", len(r.excluded), "0"),
        ("Images without a usable detection", len(r.skipped), "0"),
        (f"{r.target.point_word.capitalize()} used", r.corner_total, "0"),
        ("fx, px", r.fx, "0.0000"),
        ("fy, px", r.fy, "0.0000"),
        ("cx, px", r.cx, "0.0000"),
        ("cy, px", r.cy, "0.0000"),
    ]
    rows += [(name, float(value), "0.000000000") for name, value in zip(r.distortion_names, r.distortion)]
    if r.uncertainty is not None:
        u = r.uncertainty
        rows.append(("Point noise per coordinate, px (from the residuals)", u.noise_px, "0.0000"))
        for name in ("fx", "fy", "cx", "cy", *r.distortion_names):
            sd = u.std_of(name)
            if sd is not None:
                rows.append((f"{name}, one standard deviation", sd, "0.0000" if name in ("fx", "fy", "cx", "cy")
                             else "0.000000000"))
    rows += [
        ("Software", f"Plumbline {__version__} by Calib.Tools, OpenCV {r.opencv_version}", None),
        ("Website", "https://calib.tools", None),
    ]
    ws.append(["Calibration summary", None])
    ws["A1"].font = Font(bold=True, size=13)
    for key, value, fmt in rows:
        ws.append([key, value])
        k, v = ws.cell(row=ws.max_row, column=1), ws.cell(row=ws.max_row, column=2)
        k.font, v.font = KEY_FONT, BOLD
        v.alignment = Alignment(horizontal="left")
        if fmt:
            v.number_format = fmt
    ws["A" + str(ws.max_row + 2)] = (
        "RMS per point is sqrt(mean(dx^2 + dy^2)), the value OpenCV reports. "
        "RMS per coordinate counts x and y separately and equals it divided by sqrt(2)."
    )
    note_row = ws.max_row
    ws.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=2)
    ws.cell(row=note_row, column=1).font = KEY_FONT
    ws.cell(row=note_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[note_row].height = 32

    # ---- Images: one row per image, in file order.
    wi = wb.create_sheet("Images")
    _header(wi, ["File", "Corners", "RMS, px", "Max error, px", "Status"], [36, 10, 12, 15, 24])
    by_index = {v.index: v for v in r.views}
    for i, det in enumerate(detections):
        name = Path(det.path).name
        view = by_index.get(i)
        if view is not None:
            wi.append([name, det.count, view.rms_point, view.max_error, "used"])
        elif i in r.excluded:
            wi.append([name, det.count, error_outside_fit(r, det), None, "excluded"])
        else:
            wi.append([name, det.count, None, None, "no usable detection"])
        row = wi.max_row
        wi.cell(row=row, column=3).number_format = PX
        wi.cell(row=row, column=4).number_format = PX
        wi.cell(row=row, column=5).alignment = Alignment(horizontal="right")
    wi.auto_filter.ref = f"A1:E{wi.max_row}"

    # ---- Corners: one row per corner that took part, with where it was found and how far off.
    wc = wb.create_sheet("Corners")
    _header(wc, ["File", "Corner ID", "x, px", "y, px", "dx, px", "dy, px", "Error, px"],
            [36, 11, 12, 12, 11, 11, 11])
    for view in r.views:
        det = detections[view.index]
        name = Path(det.path).name
        for cid, (x, y), (dx, dy) in zip(det.ids.tolist(), det.corners.tolist(), view.residuals.tolist()):
            wc.append([name, int(cid), x, y, dx, dy, (dx * dx + dy * dy) ** 0.5])
    for row in wc.iter_rows(min_row=2, min_col=3, max_col=7):
        for cell in row:
            cell.number_format = PX
    wc.auto_filter.ref = f"A1:G{wc.max_row}"

    wb.properties.title = f"{project_name} calibration, revision {revision}"
    wb.properties.creator = f"Plumbline {__version__} by Calib.Tools"
    wb.save(str(path))
