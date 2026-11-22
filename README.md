# Plumbline

Camera calibration software by [Calib.Tools](https://calib.tools). Desktop app for Windows, Linux and macOS. MIT license.

**Status: in development. Installers are built for every change but are not signed or published yet.**

## What works now

- One camera at a time
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
- Export: OpenCV YAML and XML, JSON, ROS `camera_info`, Excel workbook, PDF report (summary and findings, a page with the coverage, uncertainty and residuals maps, a table of all images); choose which files to write
- Export for using the result: lookup tables that remove the lens distortion (`.npz`) and a short Python example that reads the YAML
- Boards of a set that continue the marker numbering (first marker ID)
- Calib.Tools boards by part number: pick one and the pattern settings fill in
- Project files, built-in sample images

In progress: two cameras. The solve is in `plumbline.core.rig` (how the second camera sits relative to the first, with ± on distance and angle); the window does not show it yet.

Not yet: signed installers, other wide-lens models, uncertainty, several cameras.

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
