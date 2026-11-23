"""A point misplaced on the board itself is found; a sound board raises no alarm."""
import numpy as np
import pytest

from plumbline.core import lens, synth
from plumbline.core.calibrate import calibrate
from plumbline.core.detect import Detection
from plumbline.core.findings import CHECK, findings
from plumbline.core.points import point_statistics


def _views(target, camera, poses, noise, rng, moved=None):
    """Exact projections plus pixel noise. `moved` shifts one board point: (index, (dx, dy) in mm)."""
    obj = target.object_points().astype(np.float64)
    if moved is not None:
        obj = obj.copy()
        obj[moved[0], :2] += moved[1]
    dets = []
    for i, (r, t) in enumerate(poses):
        p = lens.project(obj, r, t, camera.K, camera.dist)
        inside = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
        p = p[inside] + rng.normal(0, noise, (int(inside.sum()), 2))
        dets.append(Detection(f"v{i}.png", camera.width, camera.height,
                              np.nonzero(inside)[0].astype(np.int32), p.astype(np.float32)))
    return dets


def test_a_misplaced_point_is_found_and_measured(charuco, camera):
    poses = synth.sample_poses(charuco, camera, 24, 3)
    rng = np.random.default_rng(1)
    dets = _views(charuco, camera, poses, 0.08, rng, moved=(37, (0.30, -0.10)))
    result = calibrate(dets, charuco)
    stats = result.points
    hits = stats.suspects()
    assert [int(stats.ids[i]) for i in hits] == [37]
    # The fit gives way a little to the bad point, so slightly less than the full shift is seen.
    assert stats.offset_mm[hits[0]] == pytest.approx([0.30, -0.10], abs=0.05)
    text = next(t for tag, t in findings(result, 0.9) if tag == CHECK and "off the same way" in t)
    x, y = charuco.object_points()[37][:2]
    assert f"{x:g} mm across and {y:g} mm down" in text and "0.3" in text
    # One point is not taken for a board that is out of shape as a whole.
    assert result.board.clearly_better and not result.board.whole_board
    assert not any("fit clearly better" in t for _tag, t in findings(result, 0.9))


def test_a_sound_board_raises_no_alarm(charuco, camera, charuco_set):
    # Pure noise, several times over.
    poses = synth.sample_poses(charuco, camera, 24, 3)
    for seed in range(6):
        dets = _views(charuco, camera, poses, 0.08, np.random.default_rng(seed))
        assert calibrate(dets, charuco).points.suspects() == []
    # And the rendered sample set, which has real detection effects in it.
    _poses, _images, dets = charuco_set
    result = calibrate(dets, charuco)
    stats = point_statistics(result, dets)
    assert stats.suspects() == [] and len(stats.ids) == charuco.corner_count
    assert stats.count.max() == len(result.views)
    assert not any("off the same way" in t for _tag, t in findings(result, 0.9))


def test_points_seen_too_rarely_are_not_judged(charuco, camera):
    poses = synth.sample_poses(charuco, camera, 5, 3)
    dets = _views(charuco, camera, poses, 0.08, np.random.default_rng(2), moved=(37, (0.5, 0.0)))
    assert calibrate(dets, charuco).points.suspects() == []


def test_misplaced_point_reaches_excel_and_the_image_view(tmp_path, charuco, camera):
    from openpyxl import load_workbook

    from plumbline.core.export_xlsx import write_xlsx

    poses = synth.sample_poses(charuco, camera, 24, 3)
    dets = _views(charuco, camera, poses, 0.08, np.random.default_rng(1), moved=(37, (0.30, -0.10)))
    result = calibrate(dets, charuco)
    write_xlsx(result, dets, tmp_path / "data.xlsx", "Test", 1)
    rows = list(load_workbook(tmp_path / "data.xlsx")["Board points"].iter_rows(values_only=True))
    assert rows[0][0] == "Point ID" and len(rows) == 1 + charuco.corner_count
    noted = [row for row in rows[1:] if row[9]]
    assert [row[0] for row in noted] == [37] and noted[0][7] == pytest.approx(0.30, abs=0.05)

    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from plumbline.app import create_app
    from plumbline.ui.widgets import ImageView

    create_app([])
    view = ImageView()
    view.resize(640, 480)
    gray = np.full((camera.height, camera.width), 200, np.uint8)
    view.set_image(gray, dets[0].corners, None, "test")
    plain = view.grab().toImage()
    view.set_rings(dets[0].corners[:1])
    assert view.grab().toImage() != plain
    view.set_image(gray, dets[0].corners, None, "test")   # a new image starts without rings
    assert view.grab().toImage() == plain


def test_rows_and_columns_no_image_shows(charuco, camera):
    """A whole line of points missing from every image is said, with the edge it is on."""
    from plumbline.core import lens, synth
    from plumbline.core.calibrate import calibrate
    from plumbline.core.detect import Detection
    from plumbline.core.findings import CHECK, findings, never_seen, unseen_lines

    rng = np.random.default_rng(5)
    board = charuco.object_points().astype(np.float64)
    cols = charuco.inner_cols

    def views(keep):
        out = []
        for i, (r, t) in enumerate(synth.sample_poses(charuco, camera, 10, 7)):
            p = lens.project(board, r, t, camera.K, camera.dist, False)
            ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5) & keep
            out.append(Detection(f"v{i}.png", camera.width, camera.height, np.nonzero(ok)[0].astype(np.int32),
                                 (p[ok] + rng.normal(0, 0.05, (int(ok.sum()), 2))).astype(np.float32)))
        return out

    everything = calibrate(views(np.ones(len(board), bool)), charuco)
    assert unseen_lines(everything) == {} and never_seen(everything) is None
    column = np.arange(len(board)) % cols
    row = np.arange(len(board)) // cols
    cut = calibrate(views((column < cols - 2) & (row > 0)), charuco)
    assert unseen_lines(cut) == {"top": 1, "right": 2}
    said = [text for tag, text in findings(cut, 0.8) if tag == CHECK and "No image shows" in text]
    assert len(said) == 1 and "the row nearest the top edge and the 2 columns nearest the right edge" in said[0]
    hole = calibrate(views(row != 3), charuco)
    assert unseen_lines(hole) == {"inside": 1} and "inside the board" in never_seen(hole)


def test_a_few_far_off_points_that_steer_the_result(charuco, camera):
    """A trial without the far-off points; said only when it moves the camera by more than its stated uncertainty."""
    from plumbline.core import lens, synth
    from plumbline.core.calibrate import calibrate
    from plumbline.core.detect import Detection
    from plumbline.core.findings import CHECK, far_points, findings

    board = charuco.object_points().astype(np.float64)

    def views(seed, moved=None):
        rng = np.random.default_rng(seed)
        out = []
        for i, (r, t) in enumerate(synth.sample_poses(charuco, camera, 16, 7)):
            p = lens.project(board, r, t, camera.K, camera.dist, False)
            ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
            c = p[ok] + rng.standard_t(6, (int(ok.sum()), 2)) * 0.05      # longer tails than a bell curve, as detection has
            if moved and i == 4:
                for k, by in moved.items():
                    c[k] += by
            out.append(Detection(f"view-{i + 1:02d}.png", camera.width, camera.height, np.nonzero(ok)[0].astype(np.int32),
                                 c.astype(np.float32)))
        return out

    for seed in (1, 2, 3):
        clean = calibrate(views(seed), charuco)
        assert clean.far is not None and clean.far.dropped <= 3 and not clean.far.matters and far_points(clean) is None
    # Two corners of one image found in the wrong place: the look finds them, but they move the camera by less
    # than its stated uncertainty, so nothing is said.
    mild = calibrate(views(1, {10: (3.0, -2.5), 30: (-2.0, 2.0)}), charuco)
    assert mild.far.by_view[4] >= 2 and not mild.far.matters and far_points(mild) is None
    # Four corners in a row found 6 px off: they drag that image's whole fit with them, and it is said.
    r = calibrate(views(1, {k: (5.0, -4.0) for k in range(4)}), charuco)
    assert r.far.matters and max(r.far.by_view, key=r.far.by_view.get) == 4
    assert "mostly in view-05." in far_points(r) and "includes all points" in far_points(r)
    # That image already stands above the rest and is named for it; then only what the points do is added.
    said = [text for tag, text in findings(r, 0.8) if tag == CHECK]
    assert any("view-05 stands well above" in text for text in said)
    assert sum("furthest from the fit, nearly all in that image" in text for text in said) == 1
    assert r.rms_point > 3 * calibrate(views(1), charuco).rms_point         # nothing was taken out of the result itself
    # The trial lands closer to the camera the images were made with than the reported value does.
    name = r.far.name
    true = {"fx": camera.K[0, 0], "fy": camera.K[1, 1], "cx": camera.K[0, 2], "cy": camera.K[1, 2]}[name]
    reported = {"fx": r.fx, "fy": r.fy, "cx": r.cx, "cy": r.cy}[name]
    assert abs(reported + r.far.shift - true) < abs(reported - true)
