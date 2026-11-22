"""Three or more cameras looking at the same board."""
import json

import cv2
import numpy as np
import pytest

from plumbline.core import export, lens, synth
from plumbline.core.calibrate import CalibrationError, ModelOptions
from plumbline.core.detect import Detection
from plumbline.core.findings import group_findings
from plumbline.core.group import calibrate_group
from plumbline.core.naming import group_signature
from plumbline.core.rig import calibrate_pair


def turn(x, y):
    return cv2.Rodrigues(np.deg2rad(np.array([x, y, 0.0])))[0]


# Camera 2 to the right of camera 1 and turned inwards; camera 3 to the left, higher, turned the other way.
PLACES = [(np.eye(3), np.zeros(3)), (turn(0, -4), np.array([-120.0, 2.0, 5.0])), (turn(3, 5), np.array([130.0, -60.0, 10.0]))]


def views(target, cameras, wide, count=16, seed=3, noise=0.08, hide=()):
    """Exact projections plus pixel noise; points outside an image are dropped. `hide` is (camera, set) to leave empty."""
    rng = np.random.default_rng(seed)
    board = target.object_points().astype(np.float64)
    out = [[] for _ in cameras]
    for i, (r, t) in enumerate(synth.sample_poses(target, cameras[0], count, 7)):
        R1 = cv2.Rodrigues(np.asarray(r, np.float64))[0]
        for c, (cam, (R, T)) in enumerate(zip(cameras, PLACES)):
            p = lens.project(board, cv2.Rodrigues(R @ R1)[0].ravel(), R @ np.asarray(t, np.float64).ravel() + T,
                             cam.K, cam.dist, wide[c])
            ok = (p[:, 0] > 5) & (p[:, 0] < cam.width - 5) & (p[:, 1] > 5) & (p[:, 1] < cam.height - 5)
            if (c, i) in hide:
                ok[:] = False
            out[c].append(Detection(f"cam{c + 1}-{i:02d}.png", cam.width, cam.height, np.nonzero(ok)[0].astype(np.int32),
                                    (p[ok] + rng.normal(0, noise, (int(ok.sum()), 2))).astype(np.float32)))
    return out


def within(placed, shift, angle, sigmas=3.0):
    """Off by no more than the stated spread allows, and not by much in any case."""
    return (shift < min(sigmas * np.linalg.norm(placed.std[3:]), 1.5)
            and angle < min(sigmas * np.rad2deg(np.linalg.norm(placed.std[:3])), 0.2))


def off(placed):
    R, T = PLACES[placed.camera]
    return (float(np.linalg.norm(placed.translation - T)),
            float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(placed.rotation @ R.T)[0]))))


def test_three_cameras(tmp_path, charuco, camera):
    dets = views(charuco, [camera] * 3, [False] * 3)
    full = charuco.corner_count
    assert any(len(d.ids) < full for d in dets[2])            # part of the board is out of view in some images
    g = calibrate_group(dets, charuco)
    assert [p.camera for p in g.placed] == [1, 2] and all(p.through == 0 for p in g.placed)
    assert len(g.sets) == 16 and not g.unused and g.rms_point < 0.2
    for p in g.placed:
        assert within(p, *off(p))                             # the stated spread covers what the fit is off by
        assert 0 < p.apart_std_mm < 0.4 and 0 < p.turned_std_deg < 0.1 and np.all(p.position_std > 0)
    assert g.placed[1].apart_mm == pytest.approx(np.linalg.norm(PLACES[2][1]), abs=0.5)
    d = g.to_dict()
    assert [c["camera"] for c in d["cameras"]] == [2, 3] and d["cameras"][1]["reached_through_camera"] == 1

    # The files and what is said about the result.
    export.write_group_json(g, tmp_path / "cameras.json", ["a.json", "b.json", "c.json"])
    export.write_group_yaml(g, tmp_path / "cameras.yml")
    data = json.loads((tmp_path / "cameras.json").read_text())
    assert data["camera_count"] == 3 and data["camera_files"]["camera_3"] == "c.json" and len(data["sets"]) == 16
    assert data["cameras"][1]["translation_mm"] == pytest.approx(list(g.placed[1].translation))
    fs = cv2.FileStorage(str(tmp_path / "cameras.yml"), cv2.FILE_STORAGE_READ)
    assert fs.getNode("T3").mat().ravel() == pytest.approx(g.placed[1].translation)
    assert fs.getNode("K2").mat() == pytest.approx(g.cameras[1].camera_matrix) and fs.getNode("R1").empty()
    fs.release()
    said = group_findings(g)
    assert said[0][0] == "PASS" and sum("mm from camera 1" in text for _k, text in said) == 2
    assert len(group_signature(g)) == 16


def test_a_camera_that_never_shares_a_view_with_camera_1(charuco, camera):
    hide = {(2, i) for i in range(8)} | {(0, i) for i in range(8, 16)}     # sets 1-8: cameras 1+2, sets 9-16: cameras 2+3
    g = calibrate_group(views(charuco, [camera] * 3, [False] * 3, hide=hide), charuco)
    assert g.placed[1].through == 1 and g.placed[0].through == 0
    assert all(s.cameras in ((0, 1), (1, 2)) for s in g.sets)
    assert all(within(p, *off(p)) for p in g.placed)
    assert any("first placed through camera 2" in text for _k, text in group_findings(g))


def test_two_cameras_give_what_the_pair_solve_gives(charuco, camera):
    dets = views(charuco, [camera] * 2, [False] * 2)
    g, rig = calibrate_group(dets, charuco), calibrate_pair(dets[0], dets[1], charuco)
    assert g.placed[0].translation == pytest.approx(rig.translation, abs=1e-3)
    assert g.placed[0].position_std == pytest.approx(rig.position_std, rel=0.02)
    assert g.rms_point == pytest.approx(rig.rms_point, abs=1e-4)


def test_one_fisheye_among_standard_lenses(charuco, camera, fisheye_camera):
    dets = views(charuco, [camera, camera, fisheye_camera], [False, False, True])
    g = calibrate_group(dets, charuco, [None, None, ModelOptions(fisheye=True)])
    assert g.cameras[2].fisheye and not g.cameras[0].fisheye
    assert all(within(p, *off(p)) for p in g.placed)


def test_a_camera_with_too_few_shared_sets_is_named(charuco, camera):
    # Camera 3 has six images, but only two of them in sets where another camera saw the board too.
    hide = {(2, i) for i in range(6, 16)} | {(c, i) for c in (0, 1) for i in range(2, 6)}
    with pytest.raises(CalibrationError, match="Camera 3 cannot be placed"):
        calibrate_group(views(charuco, [camera] * 3, [False] * 3, hide=hide), charuco)
    with pytest.raises(CalibrationError, match="Camera 3: Only 2 usable images"):
        calibrate_group(views(charuco, [camera] * 3, [False] * 3, hide={(2, i) for i in range(2, 16)}), charuco)
    with pytest.raises(CalibrationError, match="same number"):
        calibrate_group([[], [None]], charuco)
