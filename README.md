# Plumbline

Camera calibration software by [Calib.Tools](https://calib.tools). Desktop app for Windows, Linux and macOS. MIT license.

**Status: in development. Installers are built for every change but are not signed or published yet.**

## What works now

- One camera, two cameras, or up to eight that look at the same target
- Targets: ChArUco, checkerboard, circle grid, asymmetric circle grid, AprilGrid (Kalibr layout) and ArUco grid (OpenCV GridBoard layout)
- OpenCV standard lens model (optional k3, tangential, rational and thin prism terms)
- OpenCV fisheye lens model (k1 to k4) for wide lenses
- Corrected view: any image with the lens distortion of the result removed
- Uncertainty: one standard deviation for every fitted parameter, and their correlations, in the result panel, the report, JSON and Excel
- Board shape: a board that is not flat or not to size is noticed, and its shape can be estimated together with the camera
- Board check: a point of the board that is misplaced (damage, a board that is not flat) is found and ringed in the images
- Residuals map: what is left over after the fit, averaged across the image; a lens model that does not fit shows up as a pattern
- Uncertainty map: how well the result is pinned down across the image, next to the coverage map
- Reprojection error per image, corner coverage map, excluding images and solving again
- A whole row or column of the board that no image shows is pointed out, with the edge it is on
- Export: OpenCV YAML and XML, JSON, ROS `camera_info`, Excel workbook, PDF report (summary and findings, a page with the coverage, uncertainty and residuals maps, a table of all images); choose which files to write
- Export for using the result: lookup tables that remove the lens distortion (`.npz`) and a short Python example that reads the YAML
- Boards of a set that continue the marker numbering (first marker ID)
- Calib.Tools boards by part number: pick one and the pattern settings fill in
- Project files, built-in sample images

Two cameras: drop or add the images of both cameras together and the file names tell them apart and pair them (`left_001` / `right_001`, `cam1-07` / `cam2-07`, `0007_0` / `0007_1`, or two folders with the same names inside). A second camera can also be added by hand and given its images. The result shows how camera 2 sits relative to camera 1 (distance, angle, position, each with ±) and each camera's own parameters. The cameras share one set of lens model settings unless camera 2 is given its own (Model tab), for example a standard lens next to a fisheye. Export writes each chosen file once per camera (`…-cam1-camera.yml`, `…-cam2-camera.yml`, …) plus `…-cameras.json`, `…-cameras.yml` and, with the report ticked, `…-cameras-report.pdf` with how camera 2 sits relative to camera 1; with the lookup tables ticked also `…-cameras-rectify-maps.npz`, which redraws both images so that a point lies on the same row in both, also for a standard lens next to a fisheye, and with the example ticked `…-cameras-example.py`, a short program that lines a pair up with those tables and places points in space (`x2 = R·x1 + T`, mm; the YAML holds K1, D1, K2, D2, R, T as OpenCV's stereo functions take them).

Three or more cameras: the same way of adding images (`cam1-07`, `cam2-07`, `cam3-07`, or one folder per camera; "Add a camera" under the image list). Each camera is calibrated on its own, then one joint fit places every camera relative to camera 1, with ± for distance, angle and position. A camera does not have to share a view with camera 1: it is reached through a camera that does. The board may be partly out of view, and standard and fisheye lenses can be mixed. The table has one row per set of images taken together; next to camera 1's image is the image of the camera selected last. Export writes each chosen file once per camera plus `…-cameras.json`, `…-cameras.yml` (Kn, Dn for every camera, Rn, Tn from camera 1) and a report on all cameras. Row-aligned images and the example program are for two cameras only.

Checking a result with other images (two or more cameras): pick images of the target that took no part in the calibration, named the way the calibration's images are. The cameras measure lengths between points of the board, and these are compared with the board's own dimensions: how lengths come out in percent, the length error, and how far the measured points are from one plane. What was found in those images is kept with the project, so the check follows when a set is left out or the lens model changes, and it is written to the report and to `…-cameras.json`.

Not yet: signed installers, other wide-lens models, several boards in one calibration, projectors.

Every result so far has been checked with rendered images only. Checks with photographs of real boards are still to come.

## How results are checked

Detection and calibration are done by OpenCV; Plumbline adds the workflow, the checks and the display around it.
The test suite renders images of a target through a camera whose parameters are known
(`plumbline.core.synth`, every pixel mapped back through the lens model) and checks that the
calibration recovers that camera, for a standard lens and for a fisheye lens. Run it with `pytest`.

The error figure shown is the RMS per point, `sqrt(mean(dx² + dy²))`, which is what OpenCV reports.
AprilGrid and ArUco grid: the corners of a tag or marker are L-shaped, and blur pulls such a corner into the tag by a fraction of a pixel. Plumbline therefore measures the edges of every tag and takes each corner where two edges cross. In the tests this brings the median corner error from about 0.3 px down to about 0.02 px. The layout is checked against the drawing calib.tools publishes for one of its boards (`tests/data`).

Uncertainty: the standard deviations are checked against OpenCV's own figures and against the spread of repeated calibrations with fresh noise. The map across the image is checked the same way, cell by cell. Before the map is drawn, the part of the uncertainty that a slight turn of the whole camera would explain is taken out, as mrcal does when it compares two models; a shift of the image centre alone is not a change of the lens. They describe random noise in the detected points only. In the rendered tests the true focal length can lie two to three standard deviations away, because detection also has small errors that repeat from image to image.

Circle grids: the centre of a tilted circle's image is not exactly the image of its centre. With the circle diameter known, Plumbline computes that shift from a first result, takes it off and solves again. In the tests this brings the mean centre error from about 0.09 px (0.25 px with the fisheye lens) down to about 0.01 px.

Residuals: with a model that fits, what is left over is noise and averages out in every part of the image. Plumbline compares the cell averages with what noise alone would give and reports a pattern when they are more than three times larger and at least 0.1 px somewhere. A fisheye lens fitted with the standard model's default terms scores about eleven; fits that are right score between 0.6 and 1.5. The shape of the residual distribution is recorded but not judged yet: rendered images give no usable limit for it.

Board shape: a board that sags by a millimetre barely raises the error but moves the focal length by more than one percent in the simulated tests. After every calibration Plumbline takes a short look at whether the images would fit a shaped board clearly better, and says so; a single misplaced point is told apart from a board that is out of shape as a whole and is named as a point. With the option on, the camera and the place of every board point are solved in turn until nothing moves; the board keeps its overall place and size, which images cannot tell. In simulation a 1.4 mm sag and a 0.3 % stretch are recovered to within 0.05 mm, with part of the board out of view in some images, for both lens models. OpenCV's own method for this (`calibrateCameraRO`) gives the same answer but needs the whole board in every image and has no fisheye version. With the option on, the stated uncertainty of the camera does not include the uncertainty of the board.

Several cameras: tests place two to six cameras at known positions, also in a chain where each camera shares views only with the next and in a closed ring, and with one fisheye among standard lenses. What the fit is off by is compared with its own stated ±. OpenCV's `calibrateMultiview` was tried first and not used: it takes only views that show the whole board and gives no ±. The check with other images is tested by moving a camera 2 % further away than it is: the length error jumps.

Board points: every point of the board is followed across all images. A point that is off the same way in every image, as seen from the board, is reported with its place on the board and the size of the offset; the Excel export lists every point.

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
