# Plumbline

Camera calibration software by [Calib.Tools](https://calib.tools). Desktop app for Windows, Linux and macOS. MIT license.

**Status: in development. There is no release to download yet.**

## What works now

- One camera at a time
- Targets: ChArUco and checkerboard
- OpenCV standard lens model (optional k3, tangential, rational and thin prism terms)
- Reprojection error per image, corner coverage map, excluding images and solving again
- Export: OpenCV YAML and XML, JSON, ROS `camera_info`, PDF report; choose which files to write
- Boards of a set that continue the marker numbering (first marker ID)
- Project files, built-in sample images

Not yet: installers, more patterns, fisheye models, uncertainty, several cameras.

## How results are checked

Detection and calibration are done by OpenCV; Plumbline adds the workflow, the checks and the display around it.
The test suite renders images of a target through a camera whose parameters are known
(`plumbline.core.synth`, every pixel mapped back through the lens model) and checks that the
calibration recovers that camera. Run it with `pytest`.

The error figure shown is the RMS per point, `sqrt(mean(dx² + dy²))`, which is what OpenCV reports.
Some tools count x and y separately; that figure is smaller by a factor of √2. Exports carry both.

## Run from source

```
python -m pip install -e ".[dev]"
python -m plumbline
```

## Earlier browser version

The browser prototype (TypeScript, opencv.js) is kept on the branch `archive/browser-p1`. It is no longer developed.

## Third-party

OpenCV (Apache 2.0), Qt for Python / PySide6 (LGPL v3), NumPy (BSD), Geist typeface (SIL OFL 1.1, see `src/plumbline/assets/fonts/OFL.txt`).
