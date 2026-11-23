"""Several boards of the same kind in every image, each a view of its own."""
import cv2
import numpy as np
import pytest

from plumbline.core import lens, synth
from plumbline.core.boards import board_places, board_targets, excluded_views, image_fit, together
from plumbline.core.calibrate import calibrate
from plumbline.core.detect import Detection
from plumbline.core.targets import CHARUCO, CHECKERBOARD, Target


def rot(v):
    return cv2.Rodrigues(np.asarray(v, np.float64))[0]


# Board 2 hinged to the right edge of board 1 and folded 50 degrees towards the camera.
R_12, T_12 = rot([0, np.deg2rad(-50), 0]), np.array([225.0, 0.0, 0.0])


def two_boards(camera, count=14, seed=3, noise=0.06):
    """Exact projections of two 7 x 5 boards plus pixel noise; points outside the image are dropped."""
    board = Target(CHARUCO, 7, 5, 30.0, 22.0, "DICT_5X5_250")
    wide = Target(CHARUCO, 14, 5, 30.0, 22.0, "DICT_5X5_250")       # only to spread the poses over both boards
    pts = board.object_points().astype(np.float64)
    rng = np.random.default_rng(seed)
    per_board = [[], []]
    for i, (r, t) in enumerate(synth.sample_poses(wide, camera, count, 7)):
        R, t = rot(r), np.asarray(t, np.float64).ravel()
        for b, (Rb, tb) in enumerate(((R, t), (R @ R_12, R @ T_12 + t))):
            p = lens.project(pts, cv2.Rodrigues(Rb)[0].ravel(), tb, camera.K, camera.dist, False)
            ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
            per_board[b].append(Detection(f"view-{i + 1:02d}.png", camera.width, camera.height,
                                          np.nonzero(ok)[0].astype(np.int32),
                                          (p[ok] + rng.normal(0, noise, (int(ok.sum()), 2))).astype(np.float32)))
    return board, per_board


def test_the_boards_of_a_set_continue_the_numbering():
    board = Target(CHARUCO, 7, 5, 30.0, 22.0, "DICT_5X5_250")
    assert [t.first_marker_id for t in board_targets(board, 3)] == [0, 17, 34]
    assert board_targets(board, 1) == [board]
    with pytest.raises(ValueError, match="only has IDs"):
        board_targets(Target(CHARUCO, 11, 8, 30.0, 22.0, "DICT_5X5_50"), 2)
    with pytest.raises(ValueError, match="numbered markers"):
        board_targets(Target(CHECKERBOARD, 10, 7, 25.0), 2)


def test_two_boards_each_a_view_of_its_own(camera):
    board, per_board = two_boards(camera)
    n = len(per_board[0])
    flat = together(per_board)
    assert len(flat) == 2 * n and flat[n].path.endswith("[board 2]") and flat[0].path == "view-01.png"
    one = calibrate(per_board[0], board)
    both = calibrate(flat, board, None, excluded_views({2}, n, 2))
    assert len(both.views) > len(one.views) and {2, n + 2}.isdisjoint(v.index for v in both.views)
    std = lambda r: float(r.uncertainty.std[r.uncertainty.names.index("fx")])     # noqa: E731
    assert std(both) < std(one) and abs(both.fx - camera.K[0, 0]) < 3 * std(both)

    # One row per image: both boards' points and what they are off by, in the same order.
    fit = image_fit(both, per_board, 0)
    assert fit.corner_count == per_board[0][0].count + per_board[1][0].count == len(fit.corners) == len(fit.residuals)
    assert image_fit(both, per_board, 2) is None

    # How board 2 sits to board 1, and how much the images disagree about it.
    (place,) = board_places(both, n, 2)
    assert place.board == 1 and place.images >= 8
    assert place.angle_deg == pytest.approx(50.0, abs=0.3) and np.linalg.norm(place.translation - T_12) < 0.8
    assert 0 < place.place_spread_mm < 1.5 and 0 < place.angle_spread_deg < 0.5
    assert place.to_dict()["board"] == 2


def test_two_boards_in_the_window(tmp_path):
    pytest.importorskip("PySide6")
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, QStandardPaths

    from plumbline.app import MainWindow, create_app
    from plumbline.core.project import Project

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    camera = synth.default_camera()
    first = Target(CHARUCO, 7, 5, 30.0, 22.0, "DICT_5X5_250")
    second = Target(CHARUCO, 7, 5, 30.0, 22.0, "DICT_5X5_250", first_marker_id=first.marker_count)
    wide = Target(CHARUCO, 14, 5, 30.0, 22.0, "DICT_5X5_250")
    rng = np.random.default_rng(7)
    files = []
    for i, (r, t) in enumerate(synth.sample_poses(wide, camera, 10, 7)):
        R, t = rot(r), np.asarray(t, np.float64).ravel()
        a = synth.render(first, camera, r, t, None).astype(np.float32)
        b = synth.render(second, camera, cv2.Rodrigues(R @ R_12)[0].ravel(), R @ T_12 + t, None).astype(np.float32)
        img = np.where(np.abs(b - 120) > 2, b, a)          # board 2 in front where both cover a pixel
        files.append(str(tmp_path / f"both-{i + 1:02d}.png"))
        cv2.imwrite(files[-1], np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8))

    def settle(bench):
        end, quiet = time.time() + 240, 0
        while time.time() < end and quiet <= 40:
            app.processEvents()
            time.sleep(0.02)
            quiet = 0 if bench._busy() else quiet + 1

    win = MainWindow()
    win.show()
    win.new_project()
    bench = win.bench
    bench._fill_fields(first)
    bench._target_changed()
    assert bench.boards_part.isHidden()

    # With one board per image only board 1 is looked for.
    bench.add_images(files)
    settle(bench)
    alone = bench.result
    assert alone is not None and len(alone.views) <= 10 and bench.table.item(0, 1).text().isdigit()

    # Two boards per image: found again, each board a view of its own, one row per image.
    bench.boards.setValue(2)
    assert bench.project.board_count == 2 and bench.result is None and "marker IDs 17 to 33" in bench.target_hint.text()
    bench.run_detection()
    settle(bench)
    both = bench.result
    assert both is not None and len(both.views) > len(alone.views) and bench.table.rowCount() == 10
    assert "+" in bench.table.item(0, 1).text() and "views of 2 boards" in bench.status.text()
    assert not bench.boards_part.isHidden() and not bench.opt_board.isEnabled()
    angle = bench.boards_grid.itemAtPosition(1, 1).widget().text()
    assert float(angle.split(" ")[0]) == pytest.approx(50.0, abs=0.5)
    fit = bench._view_for(0)
    assert fit.corner_count == sum(one[0].count for one in bench._per_board() if one[0].usable)

    # An image left out is left out with both boards; a camera cannot be added alongside.
    bench._toggle(0)
    settle(bench)
    n = len(bench.project.detections)
    assert {0, n}.isdisjoint(v.index for v in bench.result.views) and bench.table.item(0, 2).text() == "out"
    bench._add_camera()
    assert bench.project.camera_count == 1 and "one camera for now" in bench.status.text()

    # Saved and opened again with both boards.
    out = tmp_path / "out"
    out.mkdir()
    bench.export_to(str(out), formats={"json"})
    name = bench.project.name
    import json

    said = json.loads((out / f"{name}-camera.json").read_text())["boards"]
    assert said["per_image"] == 2 and said["seen_from_board_1"][0]["angle_deg"] == pytest.approx(50.0, abs=0.5)
    again = Project.load(str(out / f"{name}.plumbline"))
    assert again.board_count == 2 and len(again.board_detections) == 1 and len(again.board_detections[0]) == 10
    win.open_path(str(out / f"{name}.plumbline"))
    bench = win.bench
    settle(bench)
    assert bench.boards.value() == 2 and bench.result is not None and not bench.boards_part.isHidden()

    # The boards as one object: a tick box that shows with two boards; each further board's place gets a ± of its own.
    assert not bench.boards_fixed.isHidden() and not bench.boards_fixed.isChecked() and "tick" in bench.boards_note.text()
    free = bench.result
    bench.boards_fixed.setChecked(True)
    assert bench.project.boards_fixed and bench.result is None and "Calibrate again" in bench.status.text()
    bench.run_calibration()
    settle(bench)
    one = bench.result
    assert one is not None and one.fixed is not None and not one.fixed.loose and "as one object" in bench.status.text()
    assert [v.index for v in one.views] == [v.index for v in free.views]
    std = lambda r: float(r.uncertainty.std[r.uncertainty.names.index("fx")])     # noqa: E731
    assert std(one) < std(free) and abs(one.fx - camera.K[0, 0]) < 8
    angle = bench.boards_grid.itemAtPosition(1, 1).widget().text()
    assert float(angle.split(" ")[0]) == pytest.approx(50.0, abs=0.3) and float(angle.split("± ")[1]) < 0.2
    assert "one object" in bench.boards_note.text() and bench._view_for(1).corner_count > 0
    bench.export_to(str(out), formats={"json"})
    newest = lambda pattern: max(out.glob(pattern), key=lambda f: f.stat().st_mtime_ns)     # noqa: E731
    said = json.loads(newest("*-camera.json").read_text())["boards"]
    assert said["fixed_to_each_other"] is True and said["fit"]["loose"] is False
    assert said["seen_from_board_1"][0]["place_std_mm"] < 1.0
    assert Project.load(str(newest("*.plumbline"))).boards_fixed is True
    bench.boards_fixed.setChecked(False)
    assert not bench.project.boards_fixed and bench.result is None

    # Back to one board: what was found goes, as with any change of the target.
    bench.boards.setValue(1)
    assert bench.boards_fixed.isHidden()
    assert bench.project.board_count == 1 and not bench.project.board_detections and bench.result is None
    QSettings().setValue("project_counter", 0)


def test_three_boards_each_seen_from_board_1(camera):
    """Board 2 folded forward on the right of board 1, board 3 folded forward on its left."""
    board = Target(CHARUCO, 7, 5, 30.0, 22.0, "DICT_5X5_250")
    wide = Target(CHARUCO, 21, 5, 30.0, 22.0, "DICT_5X5_250")       # board 3, board 1, board 2 laid flat: for the poses only
    pts = board.object_points().astype(np.float64)
    R_13 = rot([0, np.deg2rad(40), 0])
    T_13 = -R_13 @ np.array([225.0, 0.0, 0.0])                     # its right edge 15 mm left of board 1
    rng = np.random.default_rng(5)
    per_board = [[], [], []]
    for i, (r, t) in enumerate(synth.sample_poses(wide, camera, 16, 11)):
        R = rot(r)
        t = R @ np.array([210.0, 0.0, 0.0]) + np.asarray(t, np.float64).ravel()      # board 1 is the middle third
        for b, (Rb, tb) in enumerate(((R, t), (R @ R_12, R @ T_12 + t), (R @ R_13, R @ T_13 + t))):
            p = lens.project(pts, cv2.Rodrigues(Rb)[0].ravel(), tb, camera.K, camera.dist, False)
            ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
            per_board[b].append(Detection(f"view-{i + 1:02d}.png", camera.width, camera.height,
                                          np.nonzero(ok)[0].astype(np.int32),
                                          (p[ok] + rng.normal(0, 0.06, (int(ok.sum()), 2))).astype(np.float32)))
    n = len(per_board[0])
    one = calibrate(per_board[0], board)
    two = calibrate(together(per_board[:2]), board)
    three = calibrate(together(per_board), board)
    std = lambda r: float(r.uncertainty.std[r.uncertainty.names.index("fx")])     # noqa: E731
    print("views", len(one.views), len(two.views), len(three.views), "fx", one.fx, two.fx, three.fx,
          "std", std(one), std(two), std(three))
    assert len(three.views) > len(two.views) > len(one.views)
    assert std(three) < std(two) < std(one) and abs(three.fx - camera.K[0, 0]) < 3 * std(three)
    assert together(per_board)[2 * n].path.endswith("[board 3]")
    second, third = board_places(three, n, 3)
    print("board 2", second.images, second.angle_deg, second.translation, "board 3", third.images, third.angle_deg, third.translation)
    assert (second.board, third.board) == (1, 2) and second.images >= 6 and third.images >= 6
    assert second.angle_deg == pytest.approx(50.0, abs=0.4) and np.linalg.norm(second.translation - T_12) < 1.0
    assert third.angle_deg == pytest.approx(40.0, abs=0.4) and np.linalg.norm(third.translation - T_13) < 1.0
    # An image left out is left out with all three boards.
    assert excluded_views({1}, n, 3) == {1, n + 1, 2 * n + 1}


def grid_pair(kind, fold_deg, gap_mm, count=10):
    """Rendered images of two 4 x 3 marker grids, board 2 to the right of board 1; returns boards, what was found, where board 2 is."""
    from plumbline.core.detect import detect
    from plumbline.core.targets import APRILGRID, ARUCO_GRID

    camera = synth.default_camera()
    code, dictionary = (APRILGRID, "DICT_APRILTAG_36h11") if kind == "aprilgrid" else (ARUCO_GRID, "DICT_5X5_250")
    first, wide = Target(code, 4, 3, 40.0, 12.0, dictionary), Target(code, 8, 3, 40.0, 12.0, dictionary)
    boards = board_targets(first, 2)
    assert boards[1].first_marker_id == 12
    fold, T = rot([0, np.deg2rad(-fold_deg), 0]), np.array([4 * 52.0 + gap_mm, 0.0, 0.0])
    rng = np.random.default_rng(9)
    per_board = [[], []]
    for i, (r, t) in enumerate(synth.sample_poses(wide, camera, count, 7)):
        R, t = rot(r), np.asarray(t, np.float64).ravel()
        a = synth.render(boards[0], camera, r, t, None).astype(np.float32)
        b = synth.render(boards[1], camera, cv2.Rodrigues(R @ fold)[0].ravel(), R @ T + t, None).astype(np.float32)
        img = np.where(np.abs(b - 120) > 2, b, a)          # board 2 in front where both cover a pixel
        img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        for k in range(2):
            per_board[k].append(detect(img, boards[k], f"both-{i + 1:02d}.png"))
    return camera, boards, per_board, T


@pytest.mark.parametrize("kind", ["aprilgrid", "aruco_grid"])
def test_two_marker_grids_found_in_rendered_images(kind):
    """Each board is found by its own numbers in an image that shows both, and neither takes the other's markers."""
    camera, boards, per_board, T = grid_pair(kind, 20.0, 60.0)
    usable = [sum(d.usable for d in dets) for dets in per_board]
    print(kind, "usable", usable, "points", [sum(d.count for d in dets) for dets in per_board])
    assert min(usable) >= 8
    # Corner numbers stay inside one board: 4 corners for each of its 12 markers.
    assert all(d.ids.max() < 48 for dets in per_board for d in dets if d.count)
    result = calibrate(together(per_board), boards[0])
    (place,) = board_places(result, 10, 2)
    print(kind, "fx", result.fx, "rms", result.rms_point, "angle", place.angle_deg, "at", place.translation, "from", place.images,
          "far", result.far.matters if result.far else None)
    assert result.rms_point < 0.2 and abs(result.fx - camera.K[0, 0]) < 12
    assert place.images >= 8 and place.angle_deg == pytest.approx(20.0, abs=0.6) and np.linalg.norm(place.translation - T) < 1.5
    assert result.far is None or not result.far.matters


@pytest.mark.parametrize("kind", ["aprilgrid", "aruco_grid"])
def test_a_board_in_front_of_another_hides_part_of_a_marker(kind):
    """Board 2 folded towards the camera covers the edge of board 1 in some images.

    A marker that is partly covered is still read, with corners a few pixels out. Nothing at
    detection can tell (in a wide lens the neighbours disagree by more than that anyway); the
    calibration can, and says so.
    """
    from plumbline.core.findings import far_points

    camera, boards, per_board, _T = grid_pair(kind, 50.0, 10.0)
    result = calibrate(together(per_board), boards[0])
    print(kind, "rms", result.rms_point, "fx", result.fx, "far", result.far and (result.far.dropped, dict(result.far.by_view),
                                                                               result.far.shift, result.far.std, result.far.rms_point))
    assert result.rms_point > 0.3
    assert result.far is not None and result.far.matters and result.far.rms_point < 0.12
    assert 4 in result.far.by_view                         # image 5, board 1: its right-hand column is behind board 2
    said = far_points(result)
    print([f.text for f in said] if isinstance(said, list) else said)
    assert said


def test_several_boards_with_several_cameras(camera):
    """Each board in each set of images is a set of its own: the fit for several cameras takes them as they come."""
    from plumbline.core.group import calibrate_group
    from plumbline.core.rig import calibrate_pair
    from test_group import PLACES, off

    board = Target(CHARUCO, 7, 5, 30.0, 22.0, "DICT_5X5_250")
    wide = Target(CHARUCO, 14, 5, 30.0, 22.0, "DICT_5X5_250")
    pts = board.object_points().astype(np.float64)
    rng = np.random.default_rng(4)
    per_camera = [[[], []] for _ in range(3)]            # [camera][board][set]
    for i, (r, t) in enumerate(synth.sample_poses(wide, camera, 12, 5)):
        R1, t1 = rot(r), np.asarray(t, np.float64).ravel()
        for c, (Rc, Tc) in enumerate(PLACES[:3]):
            for b, (Rb, tb) in enumerate(((np.eye(3), np.zeros(3)), (R_12, T_12))):
                p = lens.project(pts @ Rb.T + tb, cv2.Rodrigues(Rc @ R1)[0].ravel(), Rc @ t1 + Tc, camera.K, camera.dist, False)
                ok = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
                per_camera[c][b].append(Detection(f"cam{c + 1}-{i:02d}.png", camera.width, camera.height,
                                                  np.nonzero(ok)[0].astype(np.int32),
                                                  (p[ok] + rng.normal(0, 0.08, (int(ok.sum()), 2))).astype(np.float32)))
    one = calibrate_group([boards[0] for boards in per_camera], board)
    both = calibrate_group([together(boards) for boards in per_camera], board)
    for name, g in (("one board", one), ("two boards", both)):
        print(name, "sets", len(g.sets), [(round(off(p)[0], 3), round(off(p)[1], 4), round(float(np.linalg.norm(p.std[3:])), 3))
                                          for p in g.placed])
    assert len(both.sets) > len(one.sets)
    for a, b in zip(one.placed, both.placed):
        shift, angle = off(b)
        assert shift < 2.5 * np.linalg.norm(b.std[3:]) and angle < 2.5 * np.rad2deg(np.linalg.norm(b.std[:3]))
        assert np.linalg.norm(b.std[3:]) < 0.8 * np.linalg.norm(a.std[3:])       # more to go on: each camera is placed more firmly
    # Two cameras the same way.
    pair = calibrate_pair(together(per_camera[0]), together(per_camera[1]), board)
    assert np.linalg.norm(pair.translation - PLACES[1][1]) < 3.0
