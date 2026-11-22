"""Files that help put a result to use: lookup tables and a script to start from."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import __version__
from . import lens
from .calibrate import CalibrationResult


def write_maps(result: CalibrationResult, path: str | Path) -> None:
    """Lookup tables that remove the lens distortion, as a NumPy .npz file.

    map_x and map_y have the size of the image. For every pixel of the
    corrected image they give the place to read in the original image.
    new_camera_matrix is the pinhole camera the corrected image belongs to.
    """
    size = (result.image_width, result.image_height)
    map_x, map_y, new_K = lens.correction_maps(result.camera_matrix, result.distortion, size, result.fisheye)
    np.savez_compressed(
        str(path),
        map_x=map_x,
        map_y=map_y,
        new_camera_matrix=new_K,
        camera_matrix=result.camera_matrix,
        distortion_coefficients=result.distortion,
        image_size=np.array(size, np.int32),
        distortion_model=np.array("opencv_fisheye" if result.fisheye else "opencv"),
        generator=np.array(f"Plumbline {__version__} by Calib.Tools, https://calib.tools"),
    )


def example_script(result: CalibrationResult, camera_file: str, maps_file: str | None = None) -> str:
    """A short Python program that reads the exported camera file and uses it."""
    model = "fisheye" if result.fisheye else "standard"
    maps_note = ""
    if maps_file:
        maps_note = f"""

def load_maps(path=HERE / "{maps_file}"):
    \"\"\"The ready-made lookup tables: corrected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR).\"\"\"
    data = np.load(str(path))
    return data["map_x"], data["map_y"], data["new_camera_matrix"]
"""
    return f'''"""Use this camera calibration in your own program.

Written by Plumbline {__version__} by Calib.Tools, https://calib.tools
Camera file: {camera_file}
Lens model:  OpenCV {model} model, image {result.image_width} x {result.image_height}

Needs:  pip install opencv-python numpy
Try it: python {{name}} some_image.png      (writes some_image-corrected.png)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAMERA_FILE = HERE / "{camera_file}"


def load_camera(path=CAMERA_FILE):
    """Read the camera matrix K, the lens coefficients and the image size."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit(f"Cannot open {{path}}")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat().reshape(-1)
    size = (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real()))
    fisheye = fs.getNode("distortion_model").string() == "opencv_fisheye"
    fs.release()
    return K, dist, size, fisheye


def correction_maps(K, dist, size, fisheye):
    """Lookup tables that remove the lens distortion, and the camera matrix of the corrected image."""
    if fisheye:
        D = dist.reshape(4, 1)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=0.0)
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 0.0, size)
        map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_32FC1)
    return map_x, map_y, new_K


def correct_points(points, K, dist, fisheye, new_K):
    """Move pixel positions (N, 2) from the original image into the corrected image."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    if fisheye:
        out = cv2.fisheye.undistortPoints(pts, K, dist.reshape(4, 1), P=new_K)
    else:
        out = cv2.undistortPoints(pts, K, dist, P=new_K)
    return out.reshape(-1, 2)


def project(points_3d, rvec, tvec, K, dist, fisheye):
    """Where 3D points (N, 3) appear in the original image, for a pose rvec, tvec."""
    obj = np.asarray(points_3d, np.float64).reshape(-1, 1, 3)
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)
    if fisheye:
        out, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, dist.reshape(4, 1))
    else:
        out, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return out.reshape(-1, 2)
{maps_note}

if __name__ == "__main__":
    K, dist, size, fisheye = load_camera()
    print("Image size:", size[0], "x", size[1])
    print("Lens model:", "OpenCV fisheye" if fisheye else "OpenCV standard")
    print("Camera matrix:")
    print(K)
    print("Lens coefficients:", dist)
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit(f"Cannot read {{source}}")
        if (image.shape[1], image.shape[0]) != size:
            raise SystemExit(f"{{source}} is {{image.shape[1]}} x {{image.shape[0]}}; "
                             f"this calibration is for {{size[0]}} x {{size[1]}} images.")
        map_x, map_y, new_K = correction_maps(K, dist, size, fisheye)
        corrected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        target = source.with_name(source.stem + "-corrected.png")
        cv2.imwrite(str(target), corrected)
        print("Corrected image written to", target)
        print("Camera matrix of the corrected image:")
        print(new_K)
'''


def write_example(result: CalibrationResult, path: str | Path, camera_file: str,
                  maps_file: str | None = None) -> None:
    text = example_script(result, camera_file, maps_file).replace("{name}", Path(path).name)
    Path(path).write_text(text, encoding="utf-8")


def rig_example_script(maps_file: str, vertical: bool = False) -> str:
    """A short Python program that lines two images up with the exported tables and measures with them."""
    along, across = ("column", "row") if vertical else ("row", "column")
    return f'''"""Use this two-camera calibration in your own program.

Written by Plumbline {__version__} by Calib.Tools, https://calib.tools
Lookup tables: {maps_file}

Run it on one pair of images taken at the same moment:

    python {{name}} camera1.png camera2.png

It writes both images lined up, so that a point lies on the same {along} in both,
and one picture of the two side by side with lines drawn across to check that.
From a point's {across} in each lined-up image, point_from_pixels() gives where the
point is in space, in millimetres, seen from camera 1.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
VERTICAL = {vertical!r}   # True: the cameras sit one above the other, and points share a column


def load(path=HERE / "{maps_file}"):
    data = np.load(str(path))
    return {{key: data[key] for key in data.files}}


def line_up(image1, image2, tables=None):
    """Both images redrawn as two ideal cameras that look the same way."""
    t = tables or load()
    width, height = (int(v) for v in t["image_size"])
    for image in (image1, image2):
        if image.shape[1] != width or image.shape[0] != height:
            raise SystemExit(f"These tables are for images of {{width}} x {{height}} pixels, "
                             f"this one is {{image.shape[1]}} x {{image.shape[0]}}.")
    return (cv2.remap(image1, t["map1_x"], t["map1_y"], cv2.INTER_LINEAR),
            cv2.remap(image2, t["map2_x"], t["map2_y"], cv2.INTER_LINEAR))


def point_from_pixels(column1, row1, other, tables=None):
    """Where a point is, in mm, in the lined-up camera 1.

    column1, row1: the point in lined-up image 1. other: its {across} in lined-up image 2
    (its {along} there is the same). Arrays work too.
    """
    t = tables or load()
    shift = np.asarray(row1 if VERTICAL else column1, np.float64) - np.asarray(other, np.float64)
    pixel = np.stack(np.broadcast_arrays(np.asarray(column1, np.float64), np.asarray(row1, np.float64), shift),
                     axis=-1).reshape(-1, 1, 3)
    return cv2.perspectiveTransform(pixel, t["Q"]).reshape(-1, 3)


def distance_map(lined_up1, lined_up2, tables=None, most_shift=256):
    """Distance from camera 1 for every pixel, mm, by block matching. A starting point, not a finished matcher."""
    if VERTICAL:
        raise SystemExit("Block matching here works along rows; these cameras are lined up along columns.")
    t = tables or load()
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=most_shift, blockSize=7)
    shift = matcher.compute(lined_up1, lined_up2).astype(np.float32) / 16.0
    points = cv2.reprojectImageTo3D(shift, t["Q"])
    far = np.linalg.norm(points, axis=2)
    far[shift <= 0] = np.nan
    return far


def main(path1, path2):
    one, two = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in (path1, path2))
    if one is None or two is None:
        raise SystemExit("One of the two images could not be read.")
    a, b = line_up(one, two)
    out1, out2 = (str(Path(p).with_name(Path(p).stem + "-lined-up.png")) for p in (path1, path2))
    cv2.imwrite(out1, a)
    cv2.imwrite(out2, b)
    both = cv2.cvtColor(np.vstack([a, b]) if VERTICAL else np.hstack([a, b]), cv2.COLOR_GRAY2BGR)
    step = max(40, (both.shape[1] if VERTICAL else both.shape[0]) // 16)
    for k in range(step, both.shape[1] if VERTICAL else both.shape[0], step):
        ends = ((k, 0), (k, both.shape[0])) if VERTICAL else ((0, k), (both.shape[1], k))
        cv2.line(both, ends[0], ends[1], (18, 0, 230), 1)
    check = str(Path(path1).with_name(Path(path1).stem + "-pair-check.png"))
    cv2.imwrite(check, both)
    print(f"Written: {{out1}}, {{out2}}, {{check}}")
    print("In the check picture, every detail of the scene should sit on the same line in both halves.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Give two images: python {{name}} camera1.png camera2.png")
    main(sys.argv[1], sys.argv[2])
'''


def write_rig_example(path: str | Path, maps_file: str, vertical: bool = False) -> None:
    text = rig_example_script(maps_file, vertical).replace("{name}", Path(path).name)
    Path(path).write_text(text, encoding="utf-8")
