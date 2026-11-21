"""Circle grids: a known camera is rendered and recovered, as for the other patterns."""
import os
import time

import cv2
import numpy as np
import pytest

from plumbline.core import synth
from plumbline.core.calibrate import calibrate, projection_difference
from plumbline.core.catalog import find_board, same_geometry, stock_boards
from plumbline.core.detect import detect
from plumbline.core.findings import findings
from plumbline.core.targets import ASYMMETRIC_CIRCLES, CIRCLES, Target

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _views(target, camera, count=24, seed=7):
    poses = synth.sample_poses(target, camera, count, seed)
    rng = np.random.default_rng(seed)
    images = [synth.render(target, camera, r, t, rng) for r, t in poses]
    return poses, images, [detect(img, target, f"view-{i:02d}.png") for i, img in enumerate(images)]


@pytest.fixture(scope="module")
def asymmetric_set(camera):
    target = Target(ASYMMETRIC_CIRCLES, 12, 7, 22.0, 14.0)
    return (target, *_views(target, camera))


def test_circle_grid_geometry_matches_opencv_conventions():
    sym = Target(CIRCLES, 8, 5, 20.0, 12.0)
    assert sym.opencv_size == (8, 5) and sym.corner_count == 40
    pts = sym.object_points()
    assert tuple(pts[0]) == (0, 0, 0) and tuple(pts[1]) == (20, 0, 0) and tuple(pts[8]) == (0, 20, 0)

    asym = Target(ASYMMETRIC_CIRCLES, 8, 5, 20.0)
    assert asym.opencv_size == (4, 5) and asym.corner_count == 20
    pts = asym.object_points()
    # Rows of 4; odd rows are shifted by one lattice step.
    assert [tuple(p[:2]) for p in pts[:5]] == [(0, 0), (40, 0), (80, 0), (120, 0), (20, 20)]
    assert asym.circle_diameter_mm == pytest.approx(0.56 * 20)
    assert "5 rows of 4" in asym.describe() and sym.point_word == "circles"

    with pytest.raises(ValueError, match="even number of columns"):
        Target(ASYMMETRIC_CIRCLES, 7, 5, 20.0)
    with pytest.raises(ValueError, match="smaller than the spacing"):
        Target(CIRCLES, 8, 5, 20.0, 25.0)
    assert Target.from_dict(asym.to_dict()) == asym


@pytest.mark.parametrize("kind", [CIRCLES, ASYMMETRIC_CIRCLES])
def test_circle_grids_recover_the_camera(kind, camera):
    target = Target(kind, 12, 7, 22.0, 14.0)
    poses, _images, dets = _views(target, camera)
    used = [d for d in dets if d.usable]
    assert len(used) >= 14
    assert all(d.count == target.corner_count for d in used)  # a circle grid is found whole or not at all
    for det, (r, t) in zip(dets, poses):
        if det.usable:
            truth = synth.true_corners(target, camera, r, t)[det.ids]
            # The centre of a tilted circle's image is not exactly the image of its centre;
            # that bias is not corrected yet and stays below a quarter of a pixel here.
            assert np.max(np.linalg.norm(truth - det.corners, axis=1)) < 0.35

    result = calibrate(dets, target)
    assert abs(result.fx - 1500) / 1500 < 0.002 and abs(result.fy - 1500) / 1500 < 0.002
    assert abs(result.cx - 640) < 1.0 and abs(result.cy - 480) < 1.0
    assert result.rms_point < 0.1
    mean, _ = projection_difference(result.camera_matrix, result.distortion, camera.K, camera.dist,
                                    camera.width, camera.height)
    assert mean < 0.3
    assert any(text.startswith("Circles cover") for _tag, text in findings(result, 0.8))


def test_a_grid_that_is_partly_outside_says_so(asymmetric_set):
    target, _poses, _images, dets = asymmetric_set
    missing = [d for d in dets if not d.usable]
    assert missing, "the sample poses should push some views past the image edge"
    assert all("no complete grid of 42 circles" in d.note for d in missing)
    # The wrong counts are refused rather than matched to something else.
    good = next(i for i, d in enumerate(dets) if d.usable)
    wrong = detect(asymmetric_set[2][good], Target(ASYMMETRIC_CIRCLES, 10, 7, 22.0))
    assert wrong.count == 0 and "check the row and column counts" in wrong.note


def test_stock_circle_grids_are_listed():
    kinds = {b.target.kind for b in stock_boards()}
    assert {CIRCLES, ASYMMETRIC_CIRCLES} <= kinds
    board = find_board("CT-CIR-P-300X200-ASY")
    assert board is not None and board.target.opencv_size == (6, 7) and board.target.square_mm == 22
    # Typed in by hand without the diameter, it is still that board.
    assert same_geometry(board.target, Target(ASYMMETRIC_CIRCLES, 12, 7, 22.0))
    assert not same_geometry(board.target, Target(ASYMMETRIC_CIRCLES, 12, 7, 23.0))


def test_circle_grid_in_the_window(tmp_path, asymmetric_set):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    target, _poses, images, _dets = asymmetric_set
    paths = []
    for i, img in enumerate(images):
        paths.append(str(tmp_path / f"dots-{i:02d}.png"))
        cv2.imwrite(paths[-1], img)

    win = MainWindow()
    win.show()
    win.new_project()
    bench = win.bench
    assert bench.sx_label.text() == "Squares across" and not bench.marker.isHidden()
    bench.preset.setCurrentIndex(bench.preset.findData("CT-CIR-P-300X200-ASY"))
    assert bench.kind.currentData() == ASYMMETRIC_CIRCLES
    assert bench.project.target.part_number == "CT-CIR-P-300X200-ASY"
    assert (bench.sx.value(), bench.sy.value(), bench.square.value()) == (12, 7, 22.0)
    assert bench.sx_label.text() == "Columns" and bench.square_label.text() == "Lattice spacing"
    assert bench.marker.isHidden() and bench.dictionary.isHidden() and bench.first_id.isHidden()
    assert "42 circles, 7 rows of 6" in bench.target_hint.text()
    assert bench.table.horizontalHeaderItem(1).text() == "Circles"

    bench.add_images(paths)
    end = time.time() + 240
    quiet = 0
    while time.time() < end and quiet <= 40:
        app.processEvents()
        time.sleep(0.02)
        quiet = 0 if bench._busy() else quiet + 1
    r = bench.result
    assert r is not None and len(r.views) >= 14 and abs(r.fx - 1500) / 1500 < 0.005
    assert "circles" in bench.used_note.text() and "corners" not in bench.used_note.text()
    bench.table.selectRow(next(v.index for v in r.views))
    app.processEvents()
    assert "42 circles" in bench.view._caption

    bench.export_to(str(tmp_path), formats={"report", "opencv_yaml", "xlsx"})
    assert (tmp_path / f"{bench.project.name}-report.pdf").read_bytes().startswith(b"%PDF")

    # Back to a checkerboard: the labels follow.
    win.new_project()
    bench.kind.setCurrentIndex(bench.kind.findData("checkerboard"))
    assert bench.sx_label.text() == "Squares across" and bench.table.horizontalHeaderItem(1).text() == "Corners"
    QSettings().setValue("project_counter", 0)
