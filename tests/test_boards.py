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

    # Back to one board: what was found goes, as with any change of the target.
    bench.boards.setValue(1)
    assert bench.project.board_count == 1 and not bench.project.board_detections and bench.result is None
    QSettings().setValue("project_counter", 0)
