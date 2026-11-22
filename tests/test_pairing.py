"""Images of two cameras handed over together are told apart by their names."""
from plumbline.core.pairing import split_by_camera


def names(pattern_a, pattern_b, count=8, folder="/data/"):
    return [folder + pattern_a.format(i) for i in range(count)], [folder + pattern_b.format(i) for i in range(count)]


def test_usual_ways_of_naming_two_cameras():
    for pa, pb, labels in (
        ("left_{:03d}.png", "right_{:03d}.png", ("left", "right")),
        ("cam1-{:02d}.png", "cam2-{:02d}.png", ("1", "2")),
        ("{:04d}_0.jpg", "{:04d}_1.jpg", ("0", "1")),
        ("img{:02d}L.png", "img{:02d}R.png", ("l", "r")),
        ("Stereo_A_{:d}.bmp", "Stereo_B_{:d}.bmp", ("a", "b")),
    ):
        a, b = names(pa, pb)
        mixed = sorted(a + b, key=lambda p: p[::-1])          # handed over in no useful order
        split = split_by_camera(mixed)
        assert split is not None, pa
        assert (split.first, split.second, split.labels, split.alone) == (a, b, labels, []), pa
    # Two folders with the same file names inside.
    a = [f"/data/left/img_{i:02d}.png" for i in range(8)]
    b = [f"/data/right/img_{i:02d}.png" for i in range(8)]
    split = split_by_camera(b + a)
    assert (split.first, split.second, split.labels) == (a, b, ("left", "right"))
    # Numbers pair by value, also past 9 and 99.
    a, b = names("cam1-{:d}.png", "cam2-{:d}.png", 12)
    split = split_by_camera(a + b)
    assert split is not None and len(split.first) == 12
    assert all(x.replace("cam1", "cam2") == y for x, y in zip(split.first, split.second))
    assert split.first == a                                   # 2 comes before 10


def test_images_without_a_partner_are_set_aside():
    a, b = names("left_{:03d}.png", "right_{:03d}.png", 10)
    split = split_by_camera(a + b[:9])
    assert len(split.first) == len(split.second) == 9 and split.alone == [a[9]]
    assert split_by_camera(a + b[:4]) is None                 # too many without a partner: not a clear case


def test_one_camera_stays_one_camera():
    assert split_by_camera([f"/data/img_{i:03d}.png" for i in range(24)]) is None
    assert split_by_camera([f"/data/sample-{i:02d}.png" for i in range(1, 25)]) is None
    assert split_by_camera([f"/data/view-0{i}.png" for i in range(10)] + [f"/data/view-1{i}.png" for i in range(10)]) is None
    assert split_by_camera(["/data/left_1.png", "/data/right_1.png"]) is None     # too few to be sure
    assert split_by_camera([f"/d/IMG_{2000 + i}.JPG" for i in range(20)]) is None
