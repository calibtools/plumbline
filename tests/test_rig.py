"""Two cameras looking at the same board: where the second sits relative to the first."""
import cv2
import numpy as np
import pytest

from plumbline.core import lens, synth
from plumbline.core.calibrate import CalibrationError, ModelOptions
from plumbline.core.detect import Detection, detect
from plumbline.core.rig import calibrate_pair

TURN = np.deg2rad(4.0)
R_TRUE = cv2.Rodrigues(np.array([0.0, -TURN, 0.0]))[0]     # camera 2 turned inwards by 4 degrees
T_TRUE = np.array([-120.0, 2.0, 5.0])                      # and 120 mm to the right of camera 1


def second_pose(r, t):
    R1 = cv2.Rodrigues(np.asarray(r, np.float64))[0]
    return cv2.Rodrigues(R_TRUE @ R1)[0].ravel(), R_TRUE @ np.asarray(t, np.float64).ravel() + T_TRUE


def rendered_pairs(target, camera, count, seed):
    rng = np.random.default_rng(seed)
    first, second = [], []
    for i, (r, t) in enumerate(synth.sample_poses(target, camera, count, seed)):
        r2, t2 = second_pose(r, t)
        first.append(detect(synth.render(target, camera, r, t, rng), target, f"cam1-{i:02d}.png"))
        second.append(detect(synth.render(target, camera, r2, t2, rng), target, f"cam2-{i:02d}.png"))
    return first, second


def exact_pairs(target, camera, wide, count, seed, noise=0.08):
    """Exact projections plus pixel noise; points outside either image are dropped."""
    rng = np.random.default_rng(seed)
    board = target.object_points().astype(np.float64)
    first, second = [], []
    for i, (r, t) in enumerate(synth.sample_poses(target, camera, count, 7)):
        for store, (rr, tt), name in ((first, (r, t), "cam1"), (second, second_pose(r, t), "cam2")):
            p = lens.project(board, rr, tt, camera.K, camera.dist, wide)
            ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
            store.append(Detection(f"{name}-{i:02d}.png", camera.width, camera.height, np.nonzero(ok)[0].astype(np.int32),
                                   (p[ok] + rng.normal(0, noise, (int(ok.sum()), 2))).astype(np.float32)))
    return first, second


def off(rig):
    angle = np.rad2deg(np.linalg.norm(cv2.Rodrigues(rig.rotation @ R_TRUE.T)[0]))
    return float(np.linalg.norm(rig.translation - T_TRUE)), float(angle)


@pytest.mark.parametrize("wide", [False, True])
def test_two_rendered_cameras(wide, charuco, camera, fisheye_camera):
    cam = fisheye_camera if wide else camera
    first, second = rendered_pairs(charuco, cam, 16, 7)
    full = charuco.corner_count
    assert any(len(d.ids) < full for d in first + second)      # the board is partly out of view in some images
    rig = calibrate_pair(first, second, charuco, ModelOptions(fisheye=wide))
    shift, angle = off(rig)
    assert len(rig.pairs) == 16 and not rig.unpaired            # every pair counts, also with part of the board missing
    assert shift < 0.25 and angle < 0.03 and rig.rms_point < 0.2
    assert rig.apart_mm == pytest.approx(np.linalg.norm(T_TRUE), abs=0.2)
    assert rig.turned_deg == pytest.approx(4.0, abs=0.03)
    assert rig.position == pytest.approx(-R_TRUE.T @ T_TRUE, abs=0.3)   # camera 2 is to the right of camera 1
    assert rig.position[0] > 100
    assert 0 < rig.apart_std_mm < 0.3 and 0 < rig.turned_std_deg < 0.1
    d = rig.to_dict()
    assert d["apart_mm"] == rig.apart_mm and len(d["pairs_used"]) == 16

    if not wide:
        # Same data through OpenCV's own function, which only takes the points both cameras saw.
        O, A, B = [], [], []
        board = charuco.object_points().astype(np.float32)
        for a, b in zip(first, second):
            common = np.intersect1d(a.ids, b.ids)
            ia = {int(k): j for j, k in enumerate(a.ids)}
            ib = {int(k): j for j, k in enumerate(b.ids)}
            O.append(board[common])
            A.append(a.corners[[ia[int(k)] for k in common]])
            B.append(b.corners[[ib[int(k)] for k in common]])
        out = cv2.stereoCalibrate(O, A, B, rig.first.camera_matrix.copy(), rig.first.distortion.copy(),
                                  rig.second.camera_matrix.copy(), rig.second.distortion.copy(),
                                  (cam.width, cam.height), flags=cv2.CALIB_FIX_INTRINSIC)
        assert np.linalg.norm(out[6].ravel() - rig.translation) < 0.1
        assert np.rad2deg(np.linalg.norm(cv2.Rodrigues(out[5] @ rig.rotation.T)[0])) < 0.01


def test_a_pair_with_one_side_missing(charuco, camera):
    first, second = exact_pairs(charuco, camera, False, 12, 1)
    second[5] = Detection("cam2-05.png", camera.width, camera.height, note="No target found")
    rig = calibrate_pair(first, second, charuco)
    assert rig.unpaired == [5] and len(rig.pairs) == 11
    assert 5 in {v.index for v in rig.first.views} and 5 not in {v.index for v in rig.second.views}
    shift, angle = off(rig)
    assert shift < 0.5 and angle < 3 * rig.turned_std_deg + 0.02     # 11 pairs with 0.08 px noise

    rig = calibrate_pair(first, second, charuco, excluded={2})
    assert rig.unpaired == [2, 5]
    with pytest.raises(CalibrationError, match="same number"):
        calibrate_pair(first, second[:-1], charuco)
    empty = [Detection(d.path, d.width, d.height) for d in second]
    with pytest.raises(CalibrationError):
        calibrate_pair(first, empty, charuco)


def test_one_bad_pair_does_not_spoil_the_start(charuco, camera):
    first, second = exact_pairs(charuco, camera, False, 12, 2)
    # In one pair the board moved between the two exposures: camera 2 sees it 15 mm further left.
    bad = second[3]
    second[3] = Detection(bad.path, bad.width, bad.height, bad.ids, bad.corners + np.float32([18.0, 0.0]))
    rig = calibrate_pair(first, second, charuco)
    worst = max(rig.pairs, key=lambda p: p.rms_point)
    assert worst.index == 3 and worst.rms_point > 3 * np.median([p.rms_point for p in rig.pairs])
    clean = calibrate_pair(first, second, charuco, excluded={3})
    assert off(clean)[0] < 0.3 and clean.rms_point < 0.12


def test_stated_spread_matches_repeats(charuco, camera):
    """The ± on distance and angle against what repeated calibrations with new noise really do."""
    apart, turned, stated = [], [], []
    for seed in range(10):
        first, second = exact_pairs(charuco, camera, False, 10, 100 + seed)
        rig = calibrate_pair(first, second, charuco)
        apart.append(rig.apart_mm); turned.append(rig.turned_deg); stated.append((rig.apart_std_mm, rig.turned_std_deg))
    stated = np.median(np.array(stated), axis=0)
    print("apart: scatter", np.std(apart, ddof=1), "stated", stated[0], " turned: scatter", np.std(turned, ddof=1), "stated", stated[1])
    # Ten repeats pin the real scatter down to about a quarter, so the comparison is a coarse one.
    assert 0.5 < stated[0] / np.std(apart, ddof=1) < 2.0
    assert 0.5 < stated[1] / np.std(turned, ddof=1) < 2.0
