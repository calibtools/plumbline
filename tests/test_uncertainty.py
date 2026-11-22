"""The stated uncertainty is checked three ways: against OpenCV, against repeated trials, and for sense."""
import cv2
import numpy as np
import pytest

from plumbline.core import lens, synth
from plumbline.core.calibrate import ModelOptions, calibrate
from plumbline.core.detect import Detection
from plumbline.core.uncertainty import fitted_names


def _noisy_views(target, camera, poses, noise, rng, fisheye=False):
    """Detections made from exact projections plus pixel noise; no rendering."""
    obj = target.object_points()
    dets = []
    for i, (r, t) in enumerate(poses):
        p = lens.project(obj, r, t, camera.K, camera.dist, fisheye)
        inside = (p[:, 0] > 5) & (p[:, 0] < camera.width - 5) & (p[:, 1] > 5) & (p[:, 1] < camera.height - 5)
        p = p[inside] + rng.normal(0, noise, (int(inside.sum()), 2))
        dets.append(Detection(f"v{i}.png", camera.width, camera.height,
                              np.nonzero(inside)[0].astype(np.int32), p.astype(np.float32)))
    return dets


def test_names_follow_the_model():
    assert fitted_names(ModelOptions(), 5) == ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
    assert fitted_names(ModelOptions(tangential=False, k3=True), 5) == ["fx", "fy", "cx", "cy", "k1", "k2", "k3"]
    assert fitted_names(ModelOptions(rational=True, k3=True), 8)[-3:] == ["k4", "k5", "k6"]
    assert fitted_names(ModelOptions(fix_aspect_ratio=True), 5)[0] == "f"
    assert fitted_names(ModelOptions(fisheye=True), 4) == ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]


def test_agrees_with_opencvs_own_figures(charuco, charuco_set):
    _poses, _images, dets = charuco_set
    result = calibrate(dets, charuco)
    u = result.uncertainty
    assert u is not None and u.names == ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]

    obj_all = charuco.object_points()
    used = [d for d in dets if d.usable]
    out = cv2.calibrateCameraExtended(
        [obj_all[d.ids].astype(np.float32) for d in used], [d.corners for d in used],
        (result.image_width, result.image_height), None, None, flags=ModelOptions().flags())
    theirs = np.asarray(out[5]).reshape(-1)[:8]  # fx fy cx cy k1 k2 p1 p2
    assert np.allclose(u.std, theirs, rtol=0.02)
    assert u.noise_px == pytest.approx(result.rms_coordinate, rel=0.05)
    assert np.allclose(np.diag(u.correlation), 1.0) and np.all(np.abs(u.correlation) <= 1.0 + 1e-9)


@pytest.mark.parametrize("wide", [False, True])
def test_matches_the_spread_of_repeated_calibrations(wide, charuco, camera, fisheye_camera):
    """Calibrate many times with fresh noise: the results scatter as much as the figures say."""
    cam = fisheye_camera if wide else camera
    poses = synth.sample_poses(charuco, cam, 20, 11)
    rng = np.random.default_rng(5)
    options = ModelOptions(fisheye=wide)
    runs = [calibrate(_noisy_views(charuco, cam, poses, 0.15, rng, wide), charuco, options) for _ in range(80)]
    names = runs[0].uncertainty.names
    said = np.mean([r.uncertainty.std for r in runs], axis=0)
    values = np.array([[r.fx, r.fy, r.cx, r.cy, *r.distortion[: len(names) - 4]] for r in runs])
    assert names[4:] == runs[0].distortion_names[: len(names) - 4]
    seen = values.std(axis=0, ddof=1)
    # 80 runs give the spread to within about 16 % (two sigma), so 30 % is a fair band.
    assert np.all(np.abs(said / seen - 1) < 0.3), dict(zip(names, np.round(said / seen, 2)))
    assert np.mean([r.uncertainty.noise_px for r in runs]) == pytest.approx(0.15, rel=0.05)


def test_fewer_images_mean_less_certainty(charuco, charuco_set):
    _poses, _images, dets = charuco_set
    full = calibrate(dets, charuco)
    few = calibrate(dets[:8], charuco)
    assert few.uncertainty.std_of("fx") > 1.3 * full.uncertainty.std_of("fx")
    assert full.uncertainty.std_of("f") is None and full.uncertainty.std_of("k3") is None
    tied = calibrate(dets, charuco, ModelOptions(fix_aspect_ratio=True))
    assert tied.uncertainty.names[0] == "f" and tied.uncertainty.std_of("f") < full.uncertainty.std_of("fx") * 1.05
    assert tied.uncertainty.std_of("fx") == tied.uncertainty.std_of("f")
    assert tied.uncertainty.std_of("fy") == pytest.approx(tied.uncertainty.std_of("f") * tied.fy / tied.fx)
    pairs = full.uncertainty.strongest_pairs(0.5)
    assert all(abs(c) >= 0.5 for _a, _b, c in pairs) and pairs == sorted(pairs, key=lambda p: -abs(p[2]))
    assert set(full.uncertainty.to_dict()) >= {"parameters", "standard_deviation", "correlation"}


def test_uncertainty_reaches_the_user(tmp_path, charuco, charuco_set):
    import json

    from plumbline.core import export
    from plumbline.core.export_xlsx import write_xlsx
    from plumbline.core.findings import CHECK, NOTE, findings

    _poses, _images, dets = charuco_set
    result = calibrate(dets, charuco)
    notes = [text for tag, text in findings(result, 0.9) if tag == NOTE]
    assert any(text.startswith("Focal length ±") and "one standard deviation" in text for text in notes)

    # When the focal length is loose, the findings say so and say what to do.
    from dataclasses import replace
    u = result.uncertainty
    loose = replace(result, uncertainty=replace(u, std=u.std * 40))
    assert any(tag == CHECK and "only pinned down" in text and "tilted more strongly" in text
               for tag, text in findings(loose, 0.9))
    assert not any("Focal length" in text for _tag, text in findings(replace(result, uncertainty=None), 0.9))

    export.write_json(result, tmp_path / "camera.json")
    data = json.loads((tmp_path / "camera.json").read_text())["uncertainty"]
    assert data["parameters"][:4] == ["fx", "fy", "cx", "cy"] and len(data["correlation"]) == 8
    assert data["standard_deviation"][0] == pytest.approx(result.uncertainty.std_of("fx"))

    write_xlsx(result, dets, tmp_path / "data.xlsx", "Test", 1)
    from openpyxl import load_workbook
    labels = [row[0].value for row in load_workbook(tmp_path / "data.xlsx")["Summary"].iter_rows()]
    assert "fx, one standard deviation" in labels and "k1, one standard deviation" in labels


def test_too_many_lens_terms_show_up_as_a_loose_image_centre(charuco, charuco_set):
    from plumbline.core.findings import CHECK, findings

    _poses, _images, dets = charuco_set
    plain = calibrate(dets, charuco)
    heavy = calibrate(dets, charuco, ModelOptions(k3=True, rational=True, thin_prism=True))
    # The fit looks just as good, but the centre is far less certain.
    assert heavy.rms_point <= plain.rms_point + 1e-6
    assert heavy.uncertainty.std_of("cx") > 5 * plain.uncertainty.std_of("cx")
    assert any(tag == CHECK and "fewer" in text for tag, text in findings(heavy, 0.9))
    assert not any("image centre" in text for _tag, text in findings(plain, 0.9))
