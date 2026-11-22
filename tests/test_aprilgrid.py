"""AprilGrid (Kalibr layout): checked against the site's own drawing and against a known camera."""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from plumbline.core import synth
from plumbline.core.calibrate import ModelOptions, calibrate, projection_difference
from plumbline.core.catalog import find_board, stock_boards
from plumbline.core.detect import detect, refine_tag_corners
from plumbline.core.targets import APRILGRID, Target

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
DATA = Path(__file__).parent / "data"
FAMILY = "DICT_APRILTAG_36h11"


def _views(target, camera, count=24, seed=7):
    poses = synth.sample_poses(target, camera, count, seed)
    rng = np.random.default_rng(seed)
    images = [synth.render(target, camera, r, t, rng) for r, t in poses]
    return poses, images, [detect(img, target, f"view-{i:02d}.png") for i, img in enumerate(images)]


@pytest.fixture(scope="module")
def grid():
    return Target(APRILGRID, 7, 5, 30.0, 9.0, FAMILY)


@pytest.fixture(scope="module")
def grid_set(grid, camera):
    return _views(grid, camera)


def test_layout_is_kalibrs(grid):
    assert grid.marker_count == 35 and grid.corner_count == 140
    assert grid.width_mm == pytest.approx(7 * 30 + 6 * 9) and grid.height_mm == pytest.approx(5 * 30 + 4 * 9)
    pts = grid.object_points()
    # Tag 0 is the bottom-left tag; its corners come bottom-right, bottom-left, top-left, top-right.
    top = 4 * 39.0
    assert [tuple(p[:2]) for p in pts[:4]] == [(30, top + 30), (0, top + 30), (0, top), (30, top)]
    assert tuple(pts[4][:2]) == (39 + 30, top + 30)  # tag 1 is to its right
    assert tuple(pts[4 * 7 + 2][:2]) == (0, 3 * 39.0)  # tag 7 is above tag 0
    assert Target.from_dict(grid.to_dict()) == grid
    with pytest.raises(ValueError, match="AprilTag family"):
        Target(APRILGRID, 7, 5, 30.0, 9.0, "DICT_5X5_250")
    with pytest.raises(ValueError, match="only has IDs 0 to 586"):
        Target(APRILGRID, 7, 5, 30.0, 9.0, FAMILY, first_marker_id=560)


def test_the_sites_own_drawing_is_read_as_laid_out():
    """The drawing calib.tools shows for CT-APR-P-200X150-T36, made by the site's code, not by Plumbline.

    It is 8 px per mm with a 4.5 mm margin; the block of tags starts 6.25 mm from the left
    edge of the board and 13.75 mm from its top.
    """
    board = find_board("CT-APR-P-200X150-T36").target
    assert (board.squares_x, board.squares_y, board.square_mm, board.marker_mm) == (6, 4, 25, 7.5)
    img = cv2.imread(str(DATA / "CT-APR-P-200X150-T36.png"), cv2.IMREAD_GRAYSCALE)
    det = detect(img, board)
    assert det.count == 96 and sorted(det.ids.tolist()) == list(range(96))
    mm = (det.corners + 0.5) / 8 - 4.5 - np.array([6.25, 13.75])
    error_px = np.linalg.norm(mm - board.object_points()[det.ids][:, :2], axis=1) * 8
    # Every tag sits where the layout says, in the corner order the layout says.
    assert error_px.max() < 0.02


def test_rendered_tags_match_the_sites_drawing():
    """Plumbline's own picture of the board, used for every other test, is the same pattern."""
    board = find_board("CT-APR-P-200X150-T36").target
    site = cv2.imread(str(DATA / "CT-APR-P-200X150-T36.png"), cv2.IMREAD_GRAYSCALE)
    x0, y0 = int((6.25 + 4.5) * 8), int((13.75 + 4.5) * 8)
    ours, _origin = synth.board_texture(board, margin_squares=0)
    h, w = ours.shape
    theirs = site[y0:y0 + h, x0:x0 + w]
    different = np.abs(ours.astype(int) - theirs.astype(int)) > 128
    assert different.mean() < 0.002  # only the anti-aliased rim of the cells


def test_edges_place_corners_better_than_corner_refinement(grid, camera, grid_set):
    poses, images, _dets = grid_set
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = 2
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    finder = cv2.aruco.ArucoDetector(grid.aruco_dictionary(), params)
    plain, refined = [], []
    for (r, t), img in list(zip(poses, images))[:6]:
        truth = synth.true_corners(grid, camera, r, t)
        corners, ids, _ = finder.detectMarkers(img)
        for c, i in zip(corners, np.asarray(ids).reshape(-1)):
            c = np.asarray(c, np.float32).reshape(4, 2)
            want = truth[4 * int(i): 4 * int(i) + 4]
            plain.append(np.linalg.norm(c - want, axis=1))
            refined.append(np.linalg.norm(refine_tag_corners(img, c) - want, axis=1))
    plain, refined = np.concatenate(plain), np.concatenate(refined)
    assert np.median(plain) > 0.2  # blur pulls an L-shaped corner into the tag
    assert np.median(refined) < 0.05 and np.percentile(refined, 99) < 0.2


@pytest.mark.parametrize("wide", [False, True])
def test_aprilgrid_recovers_the_camera(wide, grid, camera, grid_set, fisheye_camera):
    cam = fisheye_camera if wide else camera
    poses, _images, dets = _views(grid, cam) if wide else grid_set
    assert sum(d.usable for d in dets) == 24  # part of the grid in view is enough
    assert any(d.count < grid.corner_count for d in dets)
    errors = np.concatenate([np.linalg.norm(synth.true_corners(grid, cam, r, t)[d.ids] - d.corners, axis=1)
                             for d, (r, t) in zip(dets, poses)])
    assert np.median(errors) < 0.06

    result = calibrate(dets, grid, ModelOptions(fisheye=wide))
    f = cam.K[0, 0]
    assert abs(result.fx - f) / f < 0.001 and abs(result.fy - f) / f < 0.001
    assert abs(result.cx - 640) < 0.5 and abs(result.cy - 480) < 0.5
    assert result.rms_point < (0.2 if wide else 0.06)
    mean, _ = projection_difference(result.camera_matrix, result.distortion, cam.K, cam.dist,
                                    cam.width, cam.height, fisheye=wide)
    assert mean < 0.25


def test_another_board_of_a_set_is_named(grid, grid_set):
    _poses, images, _dets = grid_set
    other = Target(APRILGRID, 7, 5, 30.0, 9.0, FAMILY, first_marker_id=100)
    det = detect(images[1], other)
    assert det.count == 0 and "IDs 0 to 34" in det.note and "First tag ID" in det.note


def test_aprilgrid_in_the_window(tmp_path, grid_set):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    _poses, images, _dets = grid_set
    paths = []
    for i, img in enumerate(images[:14]):
        paths.append(str(tmp_path / f"tags-{i:02d}.png"))
        cv2.imwrite(paths[-1], img)

    win = MainWindow()
    win.show()
    win.new_project()
    bench = win.bench
    # Choosing the pattern by hand switches to the usual AprilTag family and renames the fields.
    bench.kind.setCurrentIndex(bench.kind.findData(APRILGRID))
    assert bench.dictionary.currentText() == FAMILY
    assert bench.sx_label.text() == "Tags across" and bench.marker_label.text() == "Tag gap"
    assert bench.first_id_label.text() == "First tag ID" and bench.legacy.isHidden()
    assert not bench.marker.isHidden() and not bench.dictionary.isHidden()
    bench.kind.setCurrentIndex(bench.kind.findData("charuco"))
    assert bench.dictionary.currentText() == "DICT_5X5_250" and bench.marker_label.text() == "Marker size"

    bench.preset.setCurrentIndex(bench.preset.findData("CT-APR-P-300X200-T36"))
    assert bench.project.target.part_number == "CT-APR-P-300X200-T36"
    assert (bench.sx.value(), bench.sy.value(), bench.square.value(), bench.marker.value()) == (7, 5, 30.0, 9.0)
    assert "35 tags, 140 corners" in bench.target_hint.text() and "tag IDs 0 to 34" in bench.target_hint.text()

    bench.add_images(paths)
    end = time.time() + 240
    quiet = 0
    while time.time() < end and quiet <= 40:
        app.processEvents()
        time.sleep(0.02)
        quiet = 0 if bench._busy() else quiet + 1
    r = bench.result
    assert r is not None and len(r.views) == 14 and abs(r.fx - 1500) / 1500 < 0.003
    bench.export_to(str(tmp_path), formats={"report", "json"})
    assert (tmp_path / f"{bench.project.name}-report.pdf").read_bytes().startswith(b"%PDF")
    # Editing the gap away from the stock value drops the part number.
    bench.marker.setValue(10.0)
    assert bench.project.target.part_number is None
    QSettings().setValue("project_counter", 0)


def test_stock_aprilgrids_fit_their_boards():
    """Each stock AprilGrid is narrower and lower than the board it is printed on."""
    seen = 0
    for board in stock_boards():
        if board.target.kind != APRILGRID:
            continue
        seen += 1
        width, height = (float(v) for v in board.part_number.split("-")[3].split("X"))
        assert board.target.width_mm < width and board.target.height_mm < height, board.part_number
    assert seen >= 5
    big = find_board("CT-APR-P-600X400-T36").target
    assert (big.squares_x, big.squares_y) == (10, 6)
    assert (big.width_mm, big.height_mm) == pytest.approx((571.5, 337.5))
