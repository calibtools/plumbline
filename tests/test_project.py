"""A project with several cameras is saved and opened again."""
from plumbline.core.project import Project


def test_three_cameras_are_kept(tmp_path):
    from plumbline.core.calibrate import ModelOptions
    p = Project(image_paths=["a1.png"])
    assert p.camera_count == 1 and p.add_camera() == 1 and p.add_camera() == 2 and p.camera_count == 3
    p.paths_of(1).append("b1.png"); p.paths_of(2).extend(["c1.png", "c2.png"])
    p.set_options(2, ModelOptions(fisheye=True))
    assert p.pair_count == 2 and p.options_of(2).fisheye and not p.options_of(1).fisheye
    p.save(str(tmp_path / "three.plumbline"))
    q = Project.load(str(tmp_path / "three.plumbline"))
    assert q.camera_count == 3 and q.paths_of(2) == ["c1.png", "c2.png"] and q.options_of(2).fisheye
    q.remove_last_camera()
    assert q.camera_count == 2 and q.paths_of(1) == ["b1.png"]
    q.remove_last_camera(); q.remove_last_camera()
    assert q.camera_count == 1 and not q.second_image_paths
