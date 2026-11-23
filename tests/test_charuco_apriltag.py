"""ChArUco boards whose markers are AprilTags (36h11): the T36 boards in the catalogue."""
import numpy as np
import pytest

from plumbline.core import lens, synth
from plumbline.core.calibrate import calibrate
from plumbline.core.catalog import stock_boards
from plumbline.core.detect import detect
from plumbline.core.targets import CHARUCO, Target

FAMILY = "DICT_APRILTAG_36h11"
STOCK = [b.target for b in stock_boards() if b.target.kind == CHARUCO and b.target.dictionary == FAMILY]


def test_the_catalogue_has_these_boards():
    assert len(STOCK) >= 5 and all(t.part_number.endswith("-T36") for t in STOCK)


@pytest.mark.parametrize("target", STOCK, ids=lambda t: t.part_number)
def test_a_stock_board_is_found_where_it_is(target, camera):
    """One tilted view of each board: the corners are found, and found where the board was put."""
    r, t = synth.sample_poses(target, camera, 4, 7)[1]
    d = detect(synth.render(target, camera, r, t, np.random.default_rng(1)), target)
    assert d.usable and d.count >= 0.9 * target.corner_count
    put = lens.project(target.object_points().astype(np.float64)[d.ids], r, t, camera.K, camera.dist, False)
    assert np.abs(d.corners - put).max() < 0.3


def test_calibration_with_apriltag_markers(camera):
    target = Target(CHARUCO, 11, 8, 30.0, 22.0, FAMILY)
    dets = [detect(img, target, f"v{i}.png") for i, img in enumerate(synth.render_set(target, camera, 10))]
    assert sum(d.usable for d in dets) == 10
    result = calibrate(dets, target)
    assert result.rms_point < 0.1 and result.fx == pytest.approx(camera.K[0, 0], abs=2.5)
    assert result.cx == pytest.approx(camera.K[0, 2], abs=2.5)


def test_another_dictionary_does_not_pass_for_it(camera):
    """The same board looked for with ArUco 5x5 markers, and the other way round: nothing usable, no false corners."""
    april = Target(CHARUCO, 11, 8, 30.0, 22.0, FAMILY)
    aruco = Target(CHARUCO, 11, 8, 30.0, 22.0, "DICT_5X5_250")
    r, t = synth.sample_poses(april, camera, 4, 7)[1]
    rng = np.random.default_rng(1)
    assert not detect(synth.render(april, camera, r, t, rng), aruco).usable
    assert not detect(synth.render(aruco, camera, r, t, rng), april).usable
