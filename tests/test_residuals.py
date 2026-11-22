"""A lens model that does not fit leaves a pattern; one that fits leaves noise."""
import json
import os

import numpy as np
import pytest

from plumbline.core import export
from plumbline.core.calibrate import ModelOptions, calibrate
from plumbline.core.findings import CHECK, findings
from plumbline.core.residuals import residual_pattern

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_a_fitting_model_leaves_noise(charuco, charuco_set, fisheye_set):
    for dets, options in ((charuco_set[2], ModelOptions()), (fisheye_set[2], ModelOptions(fisheye=True))):
        result = calibrate(dets, charuco, options)
        p = result.pattern
        assert p is not None and p.mean.shape == (9, 12, 2) and p.count.sum() == result.corner_total
        assert 0.3 < p.strength < 2.0 and not p.has_pattern
        assert not any("not just noise" in text for _tag, text in findings(result, 0.9))


def test_the_wrong_model_leaves_a_pattern(tmp_path, charuco, fisheye_set):
    """A fisheye lens fitted with the standard model's default terms."""
    _poses, _images, dets = fisheye_set
    result = calibrate(dets, charuco)
    p = residual_pattern(result, dets)
    assert p.has_pattern and p.strength > 5 and p.largest_px > 1.0
    text = next(t for tag, t in findings(result, 0.9) if tag == CHECK and "not just noise" in t)
    assert "fisheye model" in text and "Residuals map" in text

    # The averages point outward on one side and inward on the other, not at random:
    # neighbouring cells agree far more often than chance.
    m = p.mean
    both = np.isfinite(m[:, :-1, 0]) & np.isfinite(m[:, 1:, 0])
    agree = np.sum(m[:, :-1] * m[:, 1:], axis=2)[both] > 0
    assert agree.mean() > 0.65

    export.write_json(result, tmp_path / "camera.json")
    data = json.loads((tmp_path / "camera.json").read_text())["residual_pattern"]
    assert data["strength"] == pytest.approx(p.strength) and len(data["mean_px"]) == 9

    # More terms take the pattern away again.
    better = calibrate(dets, charuco, ModelOptions(k3=True, rational=True))
    assert not better.pattern.has_pattern


def test_residual_map_in_the_small_view(charuco, fisheye_set):
    pytest.importorskip("PySide6")
    from plumbline.app import create_app
    from plumbline.ui.widgets import CoverageMap

    create_app([])
    result = calibrate(fisheye_set[2], charuco)
    view = CoverageMap()
    view.set_grid(np.ones((9, 12), int), 1.0)
    view.set_uncertainty(result.projection.sd if result.projection is not None else None)
    covered = view.grab().toImage()
    view.set_residuals(result.pattern.mean)
    assert view.grab().toImage() == covered
    view.set_mode(False, residuals=True)
    shown = view.grab().toImage()
    assert shown != covered and "left over after the fit" in view.toolTip()
    view.set_mode(True)                       # back to the uncertainty map
    assert view.grab().toImage() != shown and "standard deviation" in view.toolTip()
    view.set_mode(False)
    assert view.grab().toImage() == covered and view.toolTip() == ""
    view.set_residuals(None)
    view.set_mode(False, residuals=True)
    view.grab()


def test_a_wrong_model_is_not_blamed_on_the_board(charuco, fisheye_set):
    from plumbline.core.points import trusted_suspects

    _poses, _images, dets = fisheye_set
    wrong = calibrate(dets, charuco)
    # Taken alone, some board points look misplaced; with the pattern known they are not reported.
    assert wrong.points.suspects() and trusted_suspects(wrong) == []
    assert not any("off the same way" in text for _tag, text in findings(wrong, 0.9))
