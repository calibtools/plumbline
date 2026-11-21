# Plumbline

Camera calibration software by [Calib.Tools](https://calib.tools). Desktop app for Windows, Linux and macOS. MIT license.

**Status: in development. Installers are built for every change but are not signed or published yet.**

## What works now

- One camera at a time
- Targets: ChArUco, checkerboard, circle grid, asymmetric circle grid and AprilGrid (Kalibr layout)
- OpenCV standard lens model (optional k3, tangential, rational and thin prism terms)
- OpenCV fisheye lens model (k1 to k4) for wide lenses
- Corrected view: any image with the lens distortion of the result removed
- Reprojection error per image, corner coverage map, excluding images and solving again
- Export: OpenCV YAML and XML, JSON, ROS `camera_info`, Excel workbook, PDF report; choose which files to write
- Export for using the result: lookup tables that remove the lens distortion (`.npz`) and a short Python example that reads the YAML
- Boards of a set that continue the marker numbering (first marker ID)
- Calib.Tools boards by part number: pick one and the pattern settings fill in
- Project files, built-in sample images

Not yet: signed installers, ArUco grids, other wide-lens models, uncertainty, several cameras.

Every result so far has been checked with rendered images only. Checks with photographs of real boards are still to come.

## How results are checked

Detection and calibration are done by OpenCV; Plumbline adds the workflow, the checks and the display around it.
The test suite renders images of a target through a camera whose parameters are known
(`plumbline.core.synth`, every pixel mapped back through the lens model) and checks that the
calibration recovers that camera, for a standard lens and for a fisheye lens. Run it with `pytest`.

The error figure shown is the RMS per point, `sqrt(mean(dx² + dy²))`, which is what OpenCV reports.
AprilGrid: the corners of a tag are L-shaped, and blur pulls such a corner into the tag by a fraction of a pixel. Plumbline therefore measures the straight edges of every tag and takes each corner where two edges cross. In the tests this brings the median corner error from about 0.3 px down to about 0.02 px. The layout is checked against the drawing calib.tools publishes for one of its boards (`tests/data`).

Circle grids: the centre of a tilted circle's image is not exactly the image of its centre. Plumbline does not correct for this yet; in the tests the effect stays below a quarter of a pixel.

Some tools count x and y separately; that figure is smaller by a factor of √2. Exports carry both.

## Run from source

```
python -m pip install -e ".[dev]"
python -m plumbline
```

## Earlier browser version

The browser prototype (TypeScript, opencv.js) is kept on the branch `archive/browser-p1`. It is no longer developed.

## Third-party

OpenCV (Apache 2.0), Qt for Python / PySide6 (LGPL v3), NumPy (BSD), openpyxl (MIT), Geist typeface (SIL OFL 1.1, see `src/plumbline/assets/fonts/OFL.txt`).
