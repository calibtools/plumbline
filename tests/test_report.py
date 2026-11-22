"""The PDF report: summary, a page of maps, then the table of images."""
import shutil
import subprocess

import pytest

from plumbline.core.calibrate import calibrate


def _text_of(pdf, page):
    out = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"],
                         capture_output=True, text=True, check=True)
    return out.stdout


@pytest.mark.parametrize("which", ["charuco_set", "fisheye_set"])
def test_report_has_a_page_of_maps(which, tmp_path, charuco, request):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from plumbline.app import create_app
    from plumbline.ui.report import ROWS_PER_PAGE, write_report

    create_app([])
    _poses, _images, dets = request.getfixturevalue(which)
    # The fisheye set with the standard model: a result whose residuals have a pattern.
    result = calibrate(dets, charuco)
    path = tmp_path / "report.pdf"
    pages = write_report(result, dets, path, "Calibration-001", 1)
    assert pages == 2 + -(-len(dets) // ROWS_PER_PAGE)
    assert path.read_bytes().startswith(b"%PDF")
    if shutil.which("pdftotext") is None:
        return
    first, second, third = (_text_of(path, n) for n in (1, 2, 3))
    assert "Findings".upper() in first.upper() and "Coverage".upper() not in first.upper().split("FINDINGS")[0]
    for word in ("COVERAGE", "UNCERTAINTY", "RESIDUALS", "Typical ±", "Largest "):
        assert word in second, word
    assert ("times what noise alone would leave" in second) == result.pattern.has_pattern
    assert result.pattern.has_pattern == (which == "fisheye_set")
    assert f"page 2 of {pages}" in second and f"page 3 of {pages}" in third and "view-00.png" in third
