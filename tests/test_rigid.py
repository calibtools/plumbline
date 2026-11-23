"""Several boards fixed to each other, solved as one object."""
import cv2
import numpy as np
import pytest

from plumbline.core import lens, synth
from plumbline.core.boards import together
from plumbline.core.calibrate import CalibrationError, ModelOptions, calibrate
from plumbline.core.detect import Detection
from plumbline.core.rigid import calibrate_fixed
from plumbline.core.targets import CHARUCO, Target
from test_boards import R_12, T_12, grid_pair, rot

BOARD = Target(CHARUCO, 7, 5, 30.0, 22.0, "DICT_5X5_250")
R_13 = rot([0, np.deg2rad(40), 0])
T_13 = -R_13 @ np.array([225.0, 0.0, 0.0])


def boards(camera, count, seed, places=((R_12, T_12),), noise=0.06, wobble=None, hide=()):
    """Exact projections plus noise. places: the boards after the first, seen from board 1.

    wobble: (degrees, mm) by which every further board is off its place, anew in every image.
    hide: (board, image) pairs that are not seen.
    """
    wide = Target(CHARUCO, 14, 5, 30.0, 22.0, "DICT_5X5_250")
    pts = BOARD.object_points().astype(np.float64)
    rng = np.random.default_rng(seed)
    per_board = [[] for _ in range(len(places) + 1)]
    for i, (r, t) in enumerate(synth.sample_poses(wide, camera, count, seed)):
        R, t = rot(r), np.asarray(t, np.float64).ravel()
        for b, (Rb, tb) in enumerate(((np.eye(3), np.zeros(3)),) + tuple(places)):
            if b and wobble:
                Rb = Rb @ rot(rng.normal(0, np.deg2rad(wobble[0]), 3))
                tb = tb + rng.normal(0, wobble[1], 3)
            p = lens.project(pts @ Rb.T + tb, cv2.Rodrigues(R)[0].ravel(), t, camera.K, camera.dist, camera.fisheye)
            ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
            if (b, i) in hide:
                ok[:] = False
            per_board[b].append(Detection(f"view-{i + 1:02d}.png", camera.width, camera.height,
                                          np.nonzero(ok)[0].astype(np.int32),
                                          (p[ok] + rng.normal(0, noise, (int(ok.sum()), 2))).astype(np.float32)))
    return per_board


def std(result, name="fx"):
    return float(result.uncertainty.std[result.uncertainty.names.index(name)])


def test_fixed_boards_pin_the_camera_down_better(camera):
    off_free, off_fixed, inside = [], [], 0
    for seed in range(1, 9):
        per_board = boards(camera, 12, seed)
        free = calibrate(together(per_board), BOARD)
        fixed = calibrate_fixed(per_board, BOARD)
        assert std(fixed) < 0.75 * std(free) and fixed.rms_point < 0.1
        assert fixed.fixed.rms_free == pytest.approx(free.rms_point) and not fixed.fixed.loose
        off_free.append(free.fx - camera.K[0, 0])
        off_fixed.append(fixed.fx - camera.K[0, 0])
        inside += abs(off_fixed[-1]) < 2 * std(fixed)
        (place,) = fixed.fixed.places
        assert place.angle_deg == pytest.approx(50.0, abs=5 * place.angle_std_deg) and 0.005 < place.angle_std_deg < 0.05
        assert np.linalg.norm(place.translation - T_12) < 5 * place.place_std_mm and 0.01 < place.place_std_mm < 0.08
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))     # noqa: E731
    print("fx off, each on its own", rms(off_free), "fixed", rms(off_fixed), "inside two ±:", inside, "of 8")
    assert rms(off_fixed) < 0.8 * rms(off_free)
    assert inside >= 6          # the stated ± is about what the result is off by


def test_the_result_reads_like_the_one_for_boards_on_their_own(camera):
    per_board = boards(camera, 12, 3)
    n = len(per_board[0])
    free = calibrate(together(per_board), BOARD, None, {2, n + 2})
    fixed = calibrate_fixed(per_board, BOARD, None, {2})
    assert [v.index for v in fixed.views] == [v.index for v in free.views] and {2, n + 2}.isdisjoint(v.index for v in fixed.views)
    assert [v.path for v in fixed.views] == [v.path for v in free.views] and fixed.excluded == free.excluded
    # Every board in every image still has a position of its own to draw with: the object's, then the board's place on it.
    by = {v.index: v for v in fixed.views}
    place = fixed.fixed.places[0]
    for i in (0, 5):
        first, second = by[i], by[n + i]
        assert np.allclose(rot(second.rvec), rot(first.rvec) @ place.rotation, atol=1e-9)
        assert np.allclose(second.tvec, rot(first.rvec) @ place.translation + first.tvec, atol=1e-9)
        det = per_board[1][i]
        shown = lens.project(BOARD.object_points()[det.ids].astype(np.float64), second.rvec, second.tvec,
                             fixed.camera_matrix, fixed.distortion, False)
        assert np.allclose(shown - det.corners, second.residuals, atol=1e-4)
    assert fixed.points is not None and fixed.uncertainty.degrees_of_freedom > free.uncertainty.degrees_of_freedom
    assert fixed.projection is not None
    assert place.to_dict()["place_std_mm"] == place.place_std_mm


def test_three_boards_as_one_object(camera):
    per_board = boards(camera, 14, 5, places=((R_12, T_12), (R_13, T_13)))
    fixed = calibrate_fixed(per_board, BOARD)
    second, third = fixed.fixed.places
    print("fx", fixed.fx, "±", std(fixed), second.angle_deg, third.angle_deg, second.place_std_mm, third.place_std_mm)
    assert abs(fixed.fx - camera.K[0, 0]) < 3 * std(fixed)
    assert second.angle_deg == pytest.approx(50.0, abs=0.15) and np.linalg.norm(second.translation - T_12) < 0.3
    assert third.angle_deg == pytest.approx(40.0, abs=0.15) and np.linalg.norm(third.translation - T_13) < 0.3


def test_boards_that_move_against_each_other_show(camera):
    """Board 2 is half a degree and a millimetre off its place, differently in every image: not one object."""
    per_board = boards(camera, 12, 4, wobble=(0.5, 1.0))
    fixed = calibrate_fixed(per_board, BOARD)
    print("each on its own", fixed.fixed.rms_free, "fixed", fixed.fixed.rms_fixed)
    assert fixed.fixed.loose and fixed.fixed.rms_fixed > 3 * fixed.fixed.rms_free
    assert fixed.fixed.to_dict()["loose"] is True
    # A tenth of that is within what the images can tell.
    still = calibrate_fixed(boards(camera, 12, 4, wobble=(0.01, 0.02)), BOARD)
    assert not still.fixed.loose


def test_an_image_without_board_1_still_counts(camera):
    hide = {(0, 1), (0, 6)}
    per_board = boards(camera, 12, 6, hide=hide)
    fixed = calibrate_fixed(per_board, BOARD)
    n = len(per_board[0])
    by = {v.index for v in fixed.views}
    assert 1 not in by and n + 1 in by and n + 6 in by
    assert abs(fixed.fx - camera.K[0, 0]) < 3 * std(fixed)


def test_a_board_never_seen_with_board_1_cannot_be_placed(camera):
    per_board = boards(camera, 12, 6)
    n = len(per_board[0])
    # Board 2 only where board 1 is not.
    hide = {(0, i) for i in range(0, n, 2)} | {(1, i) for i in range(1, n, 2)}
    with pytest.raises(CalibrationError, match="together with board 1"):
        calibrate_fixed(boards(camera, 12, 6, hide=hide), BOARD)
    with pytest.raises(CalibrationError, match="one board per image"):
        calibrate_fixed(per_board, BOARD, ModelOptions(release_board=True))
    assert calibrate_fixed(per_board[:1], BOARD).fixed is None      # one board: nothing to fix


def test_fixed_boards_with_a_fisheye_lens(fisheye_camera):
    per_board = boards(fisheye_camera, 14, 2, places=((rot([0, np.deg2rad(-30), 0]), T_12),))
    options = ModelOptions(fisheye=True)
    free = calibrate(together(per_board), BOARD, options)
    fixed = calibrate_fixed(per_board, BOARD, options)
    name = fixed.uncertainty.names[0]
    print("fisheye: each on its own", free.fx, "±", std(free, name), "fixed", fixed.fx, "±", std(fixed, name),
          "place", fixed.fixed.places[0].angle_deg)
    assert fixed.options.fisheye and std(fixed, name) < std(free, name)
    assert abs(fixed.fx - fisheye_camera.K[0, 0]) < 3 * std(fixed, name)
    assert fixed.fixed.places[0].angle_deg == pytest.approx(30.0, abs=0.2)


def test_a_partly_hidden_marker_still_shows_with_fixed_boards():
    """Rendered marker grids, board 2 in front of board 1's edge in some images (see test_boards)."""
    _camera, grids, per_board, _T = grid_pair("aruco_grid", 50.0, 10.0)
    fixed = calibrate_fixed(per_board, grids[0])
    print("rms", fixed.rms_point, "far", fixed.far and (fixed.far.dropped, dict(fixed.far.by_view), fixed.far.shift, fixed.far.std))
    assert fixed.far is not None and fixed.far.matters and 4 in fixed.far.by_view
