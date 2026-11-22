"""ArUco grid (OpenCV GridBoard layout): checked against a drawing made by the site's code and a known camera."""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from plumbline.core import synth
from plumbline.core.calibrate import calibrate, projection_difference
from plumbline.core.detect import detect
from plumbline.core.targets import ARUCO_GRID, Target

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def grid():
    return Target(ARUCO_GRID, 6, 4, 22.0, 8.0, "DICT_6X6_250", first_marker_id=10)


def test_layout_is_opencvs_gridboard(grid):
    assert grid.marker_count == 24 and grid.corner_count == 96 and grid.border_bits == 1
    assert (grid.width_mm, grid.height_mm) == (6 * 22 + 5 * 8, 4 * 22 + 3 * 8)
    ours = grid.object_points().reshape(24, 4, 3)
    board = cv2.aruco.GridBoard((6, 4), 22.0, 8.0, grid.aruco_dictionary(), np.arange(10, 34, dtype=np.int32))
    theirs = np.asarray(board.getObjPoints(), np.float32).reshape(24, 4, 3)
    assert np.allclose(ours, theirs, atol=1e-4)
    assert "ArUco grid 6 x 4 markers" in grid.describe() and "IDs from 10" in grid.describe()
    with pytest.raises(ValueError, match="marker IDs 240 to 263"):
        Target(ARUCO_GRID, 6, 4, 22.0, 8.0, "DICT_6X6_250", first_marker_id=240)


def test_a_drawing_made_by_the_site_is_read_as_laid_out(grid):
    """A 6 x 4 grid drawn by the calib.tools generator code (pitch 30 mm, markers 22 mm, IDs from 10).

    8 px per mm with a 4.5 mm margin; the block of markers starts 14 mm from the left edge
    of the 200 x 150 mm board and 19 mm from its top.
    """
    img = cv2.imread(str(DATA / "aruco-grid-6x4-30-22-from-10.png"), cv2.IMREAD_GRAYSCALE)
    det = detect(img, grid)
    assert det.count == 96 and sorted(det.ids.tolist()) == list(range(96))
    mm = (det.corners + 0.5) / 8 - 4.5 - np.array([14.0, 19.0])
    error_px = np.linalg.norm(mm - grid.object_points()[det.ids][:, :2], axis=1) * 8
    assert error_px.max() < 0.02

    ours, _origin = synth.board_texture(grid, margin_squares=0)
    h, w = ours.shape
    y0, x0 = int((19 + 4.5) * 8), int((14 + 4.5) * 8)
    theirs = img[y0:y0 + h, x0:x0 + w]
    different = np.abs(ours.astype(int) - theirs.astype(int)) > 128
    assert different.mean() < 0.002


def test_aruco_grid_recovers_the_camera(camera):
    target = Target(ARUCO_GRID, 7, 5, 30.0, 9.0, "DICT_5X5_250")
    poses = synth.sample_poses(target, camera, 24, 7)
    rng = np.random.default_rng(7)
    dets = [detect(synth.render(target, camera, r, t, rng), target, f"v{i}.png") for i, (r, t) in enumerate(poses)]
    assert sum(d.usable for d in dets) == 24
    errors = np.concatenate([np.linalg.norm(synth.true_corners(target, camera, r, t)[d.ids] - d.corners, axis=1)
                             for d, (r, t) in zip(dets, poses)])
    assert np.median(errors) < 0.06
    result = calibrate(dets, target)
    assert abs(result.fx - 1500) / 1500 < 0.001 and abs(result.fy - 1500) / 1500 < 0.001
    assert result.rms_point < 0.06
    mean, _ = projection_difference(result.camera_matrix, result.distortion, camera.K, camera.dist,
                                    camera.width, camera.height)
    assert mean < 0.25
    other = detect(synth.render(target, camera, *poses[1], rng),
                   Target(ARUCO_GRID, 7, 5, 30.0, 9.0, "DICT_5X5_250", first_marker_id=100))
    assert other.count == 0 and "First marker ID" in other.note


def test_aruco_grid_fields_in_the_window():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app

    QStandardPaths.setTestModeEnabled(True)
    create_app([])
    QSettings().clear()
    win = MainWindow()
    win.show()
    win.new_project()
    bench = win.bench
    bench.kind.setCurrentIndex(bench.kind.findData("aprilgrid"))
    bench.kind.setCurrentIndex(bench.kind.findData(ARUCO_GRID))
    assert bench.dictionary.currentText() == "DICT_5X5_250"  # back from the AprilTag family
    assert bench.sx_label.text() == "Markers across" and bench.marker_label.text() == "Marker gap"
    assert bench.first_id_label.text() == "First marker ID" and bench.legacy.isHidden()
    bench.sx.setValue(6)
    bench.sy.setValue(4)
    bench.square.setValue(22.0)
    bench.marker.setValue(8.0)
    assert bench.project.target == Target(ARUCO_GRID, 6, 4, 22.0, 8.0, "DICT_5X5_250")
    assert "24 markers, 96 corners" in bench.target_hint.text() and "marker IDs 0 to 23" in bench.target_hint.text()
    QSettings().setValue("project_counter", 0)
