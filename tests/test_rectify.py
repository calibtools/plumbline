"""Row alignment for two cameras: the same point lands on the same row, and distances come out right."""
import cv2
import numpy as np
import pytest

from plumbline.core import synth
from plumbline.core.calibrate import ModelOptions
from plumbline.core.detect import detect
from plumbline.core.rectify import rectify, write_rectify_maps
from plumbline.core.rig import calibrate_pair
from test_rig import rendered_pairs, second_pose


def aligned_points(charuco, cam1, cam2, rig, r, pose_index=3):
    """Render one more pair, redraw both images with the tables, and find the board's corners in them."""
    rv, tv = synth.sample_poses(charuco, cam1, 16, 7)[pose_index]
    rng = np.random.default_rng(99)
    one = synth.render(charuco, cam1, rv, tv, rng)
    two = synth.render(charuco, cam2, *second_pose(rv, tv), rng)
    a = detect(cv2.remap(one, *r.map1, cv2.INTER_LINEAR), charuco)
    b = detect(cv2.remap(two, *r.map2, cv2.INTER_LINEAR), charuco)
    common = np.intersect1d(a.ids, b.ids)
    ia = {int(k): j for j, k in enumerate(a.ids)}
    ib = {int(k): j for j, k in enumerate(b.ids)}
    return common, a.corners[[ia[int(k)] for k in common]], b.corners[[ib[int(k)] for k in common]]


@pytest.mark.parametrize("mixed", [False, True])
def test_points_share_a_row_and_distances_are_right(mixed, tmp_path, charuco, camera, fisheye_camera):
    cam2 = fisheye_camera if mixed else camera
    rng = np.random.default_rng(7)
    first, second = [], []
    for i, (rv, tv) in enumerate(synth.sample_poses(charuco, camera, 16, 7)):
        first.append(detect(synth.render(charuco, camera, rv, tv, rng), charuco, f"cam1-{i:02d}.png"))
        second.append(detect(synth.render(charuco, cam2, *second_pose(rv, tv), rng), charuco, f"cam2-{i:02d}.png"))
    rig = calibrate_pair(first, second, charuco, ModelOptions(), ModelOptions(fisheye=mixed))
    r = write_rectify_maps(rig, tmp_path / "maps.npz")
    assert not r.vertical and r.size == (camera.width, camera.height)

    common, pa, pb = aligned_points(charuco, camera, cam2, rig, r)
    assert len(common) >= 20
    rows = pa[:, 1] - pb[:, 1]
    print("row difference px: mean", rows.mean(), "largest", np.abs(rows).max())
    assert np.abs(rows).max() < (0.6 if mixed else 0.25)       # the wide camera is redrawn much enlarged

    # Disparity to points in space, with Q: neighbouring corners of the board are one square apart.
    disparity = pa[:, 0] - pb[:, 0]
    pts = cv2.perspectiveTransform(np.column_stack([pa, disparity]).reshape(-1, 1, 3).astype(np.float64), r.Q).reshape(-1, 3)
    board = charuco.object_points()
    apart, expected = [], []
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            d = np.linalg.norm(board[common[i]] - board[common[j]])
            if d > 100:                                         # long spans say more than neighbours
                apart.append(np.linalg.norm(pts[i] - pts[j]))
                expected.append(d)
    ratio = np.array(apart) / np.array(expected)
    print("measured / true length: median", np.median(ratio), "spread", ratio.std())
    assert np.median(ratio) == pytest.approx(1.0, abs=0.01 if mixed else 0.004)

    saved = np.load(tmp_path / "maps.npz")
    assert str(saved["aligned_along"]) == "rows" and saved["map2_x"].shape == (camera.height, camera.width)
    assert np.allclose(saved["Q"], r.Q) and np.allclose(saved["map1_y"], r.map1[1])
    again = rectify(rig)
    assert np.allclose(again.P2, r.P2)
