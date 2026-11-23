# Changes

What changed in each version of Plumbline, newest first. Versions follow the pattern `0.minor.patch` until the
first stable release.

## 0.1.0 (not yet released)

First public version.

One camera
- Targets: ChArUco, checkerboard, circle grid, asymmetric circle grid, AprilGrid (Kalibr layout), ArUco grid; Calib.Tools boards by part number
- Lens models: OpenCV standard (optional k3, tangential, rational, thin prism) and fisheye
- Error for every image; leave an image out and solve again; coverage map
- A ± for every parameter, and a map of how well the result is pinned down across the image
- Residuals map with a hint when the lens model does not fit
- A board that is not flat is noticed; its shape can be solved together with the camera
- A misplaced point of the board is found; a row or column that no image shows is pointed out
- A few far-off points that steer the result are pointed out
- Corrected view of any image

Several cameras
- Two to eight cameras, told apart and matched by file name; standard and fisheye lenses can be mixed
- Where every camera sits relative to camera 1, with ±
- Two cameras: row-aligned images and a short example program
- Check with images that took no part: lengths on the board, measured by the cameras

Several boards
- Up to four boards of the same kind in every image, each found by its own marker numbers
- Boards fixed to each other can be solved as one object

Files
- OpenCV YAML and XML, JSON, ROS `camera_info`, Excel, PDF report, lookup tables, Python example, project file
- Files are named by project and revision and never overwrite older ones

App
- Built-in samples for one camera and for two
- About window with version details and a way to report a problem

Known limits
- Checked with rendered images only; photographs of real boards are still to come
- Installers are not signed: Windows and macOS warn when the app is first opened
