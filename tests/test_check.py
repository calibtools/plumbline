"""Checking a two-camera result with pairs that took no part in it: the board as a ruler."""
import numpy as np
import pytest

from plumbline.core import synth
from plumbline.core.calibrate import ModelOptions
from plumbline.core.check import check_pairs, locate
from plumbline.core.detect import Detection, detect
from plumbline.core.rig import calibrate_pair
from test_rig import T_TRUE, exact_pairs, rendered_pairs, second_pose


def test_lengths_on_pairs_that_took_no_part(charuco, camera):
    first, second = rendered_pairs(charuco, camera, 16, 7)
    rig = calibrate_pair(first, second, charuco)
    # New poses, new noise: seed 21 gives a different set of views than the calibration used.
    held1, held2 = rendered_pairs(charuco, camera, 8, 21)
    result = check_pairs(rig, held1, held2, charuco)
    print("scale", result.scale, "length rms", result.length_error_rms_mm, "max", result.length_error_max_mm,
          "flatness", result.flatness_rms_mm, "spans", result.spans, "longest", result.longest_mm)
    assert len(result.pairs) + len(result.skipped) == 8 and len(result.pairs) >= 6
    assert result.scale == pytest.approx(1.0, abs=0.002)
    assert result.length_error_rms_mm < 0.15 and result.flatness_rms_mm < 0.15
    assert result.longest_mm > 250 and result.spans > 1000
    assert all(400 < p.distance_mm < 2500 and p.ray_gap_rms_mm < 0.2 for p in result.pairs)
    assert result.to_dict()["spans_compared"] == result.spans


def test_a_wrong_distance_between_the_cameras_shows_as_a_wrong_scale(charuco, camera):
    first, second = exact_pairs(charuco, camera, False, 12, 3)
    rig = calibrate_pair(first, second, charuco)
    held1, held2 = exact_pairs(charuco, camera, False, 12, 4)
    good = check_pairs(rig, held1[6:], held2[6:], charuco)
    assert good.scale == pytest.approx(1.0, abs=0.002)
    # The same result with the cameras taken to be 1 % further apart: every length comes out 1 % long.
    rig.translation = rig.translation * 1.01
    stretched = check_pairs(rig, held1[6:], held2[6:], charuco)
    assert stretched.scale == pytest.approx(1.01, abs=0.002)
    assert stretched.length_error_rms_mm > 5 * good.length_error_rms_mm


def test_points_are_placed_where_they_are(charuco, camera):
    first, second = exact_pairs(charuco, camera, False, 12, 5, noise=0.0)
    rig = calibrate_pair(first, second, charuco)
    r, t = synth.sample_poses(charuco, camera, 12, 7)[2]
    import cv2
    board = charuco.object_points().astype(np.float64)
    truth = (cv2.Rodrigues(np.asarray(r, np.float64))[0] @ board.T).T + np.asarray(t, np.float64).ravel()
    common = np.intersect1d(first[2].ids, second[2].ids)
    ia = {int(k): j for j, k in enumerate(first[2].ids)}
    ib = {int(k): j for j, k in enumerate(second[2].ids)}
    pts, gap = locate(rig, first[2].corners[[ia[int(k)] for k in common]], second[2].corners[[ib[int(k)] for k in common]])
    assert np.abs(pts - truth[common]).max() < 0.05 and gap.max() < 0.01
    with pytest.raises(ValueError):
        check_pairs(rig, [Detection("a", 10, 10)], [Detection("b", 10, 10)], charuco)
