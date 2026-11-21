import numpy as np
import pytest

from plumbline.core import synth
from plumbline.core.detect import detect
from plumbline.core.targets import CHARUCO, CHECKERBOARD, Target


@pytest.fixture(scope="session")
def charuco():
    return Target(CHARUCO, 11, 8, 30.0, 22.0, "DICT_5X5_250")


@pytest.fixture(scope="session")
def checkerboard():
    return Target(CHECKERBOARD, 10, 7, 25.0)


@pytest.fixture(scope="session")
def camera():
    return synth.default_camera()


def _make(target, camera, count, seed):
    poses = synth.sample_poses(target, camera, count, seed)
    rng = np.random.default_rng(seed)
    images = [synth.render(target, camera, r, t, rng) for r, t in poses]
    dets = [detect(img, target, f"view-{i:02d}.png") for i, img in enumerate(images)]
    return poses, images, dets


@pytest.fixture(scope="session")
def charuco_set(charuco, camera):
    return _make(charuco, camera, 24, 7)


@pytest.fixture(scope="session")
def checker_set(checkerboard, camera):
    return _make(checkerboard, camera, 20, 11)


@pytest.fixture(scope="session")
def fisheye_camera():
    return synth.fisheye_camera()


@pytest.fixture(scope="session")
def fisheye_set(charuco, fisheye_camera):
    return _make(charuco, fisheye_camera, 24, 7)
