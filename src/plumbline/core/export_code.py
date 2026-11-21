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
