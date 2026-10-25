# OpenCV.js Feasibility Report

## Version & Source

- **Version**: OpenCV.js **4.9.0** (official pre-built release)
- **Source**: `https://docs.opencv.org/4.9.0/opencv.js`
- **Type**: Single-file asm.js build (~10 MB), no external `.wasm` dependency
- **Build modules**: calib3d, core, dnn, features2d, flann, imgproc, js, objdetect, photo, video
- **Custom build**: Not required. The official build includes all needed ChArUco APIs.

## Available APIs

### Working (verified in-browser via Playwright)

| API | Signature Notes |
|-----|----------------|
| `cv.getPredefinedDictionary(dictId)` | Free function. Dict IDs are on `cv` directly (e.g., `cv.DICT_4X4_50 = 0`), **not** on `cv.aruco`. |
| `cv.aruco_CharucoBoard(size, squareLen, markerLen, dict, ids)` | Constructor requires **5 params**. Pass `new cv.Mat()` for default IDs. |
| `board.generateImage(outSize, outMat, margin, borderBits)` | Works. |
| `board.setLegacyPattern(bool)` / `board.getLegacyPattern()` | Works. |
| `cv.aruco_CharucoParameters()` | Zero-arg constructor. |
| `cv.aruco_DetectorParameters()` | Zero-arg constructor. |
| `cv.aruco_RefineParameters(minRepDist, errCorrRate, checkAllOrders)` | Requires **3 params** (no zero-arg ctor). Defaults: `(10, 3, true)`. |
| `cv.aruco_CharucoDetector(board, charucoParams, detectorParams, refineParams)` | Requires exactly **4 params**. |
| `detector.detectBoard(img, charucoCorners, charucoIds, markerCorners, markerIds)` | Works. Returns corners as `Nx1 CV_32FC2`, IDs as `Nx1 CV_32SC1`. |
| `cv.calibrateCameraExtended(objPtsVec, imgPtsVec, imgSize, camMat, distCoeffs, rvecs, tvecs, stdDevIntr, stdDevExtr, perViewErr, flags, criteria)` | Full extended signature with per-view errors and std devs. |
| `cv.solvePnP(objPts, imgPts, camMat, distCoeffs, rvec, tvec)` | Works. |
| `cv.Rodrigues(rvec, rotMat)` | Works. |
| `cv.cvtColor`, `cv.resize`, `cv.GaussianBlur`, `cv.threshold`, `cv.goodFeaturesToTrack` | Standard imgproc functions. |
| `cv.drawDetectedCornersCharuco(img, corners, ids, color)` | Visualization helper. |

### Broken / Missing (with workarounds)

| API | Issue | Workaround |
|-----|-------|------------|
| `board.matchImagePoints(corners, ids, objPts, imgPts)` | Unbound `vector<Point3f>` type in JS bindings. Returns undefined or throws. | **Manual computation**: For detected corner ID `i`, object point is `((i % innerX + 1) * squareLen, (floor(i / innerX) + 1) * squareLen, 0)` where `innerX = squaresX - 1`. |
| `board.getChessboardCorners()` | Same unbound type issue. | Same as above — compute from board geometry. |
| `cv.cornerSubPix(...)` | Not exposed in the official JS build. | `CharucoDetector` performs internal sub-pixel refinement (controlled by `detectorParams.cornerRefinementMethod`). For the two-pass detection flow (detect at reduced res, refine at full res), we run the full `CharucoDetector.detectBoard` on both resolutions. |
| `cv.calibrateCamera(...)` | Not exposed. | Use `cv.calibrateCameraExtended(...)` which is a superset. |
| `cv.aruco` namespace | Does not exist as an object. | Dictionary constants are on `cv` directly: `cv.DICT_4X4_50`, `cv.DICT_6X6_250`, etc. |

### Dictionary Constants

All predefined dictionary constants are available directly on `cv`:

```
DICT_4X4_50 (0), DICT_4X4_100 (1), DICT_4X4_250 (2), DICT_4X4_1000 (3),
DICT_5X5_50 (4), DICT_5X5_100 (5), DICT_5X5_250 (6), DICT_5X5_1000 (7),
DICT_6X6_50 (8), DICT_6X6_100 (9), DICT_6X6_250 (10), DICT_6X6_1000 (11),
DICT_7X7_50 (12), DICT_7X7_100 (13), DICT_7X7_250 (14), DICT_7X7_1000 (15),
DICT_ARUCO_ORIGINAL (16),
DICT_APRILTAG_16h5 (17), DICT_APRILTAG_25h9 (18),
DICT_APRILTAG_36h10 (19), DICT_APRILTAG_36h11 (20),
DICT_ARUCO_MIP_36h12 (21)
```

## Verification

Tested 2026-10-25 using Playwright (headless Chromium). Full pipeline verified:
1. Created `DICT_4X4_50` dictionary → 50 markers, 4-bit, 2-byte bytesList
2. Created 5×7 CharucoBoard (sq=30mm, mk=22mm) → 700×500 rendered image
3. `detectBoard` found 24/24 charuco corners and 17 markers
4. Manual point matching → `calibrateCameraExtended` → RMS 0.004 (single frontal image)
5. `solvePnP` + `Rodrigues` → valid rotation/translation

## Decision

**Use the official opencv.js 4.9.0 build.** No custom Emscripten build needed. The missing `cornerSubPix` and `matchImagePoints` bindings are worked around without loss of accuracy since `CharucoDetector` handles sub-pixel refinement internally, and object point computation from board geometry is trivial and exact.
