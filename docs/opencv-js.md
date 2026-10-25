# OpenCV.js Build Report

## Version & Source

- **Version**: OpenCV.js **4.9.0** (official pre-built release)
- **Source**: `https://docs.opencv.org/4.9.0/opencv.js`
- **Type**: Single-file **WASM** build (~10 MB). Contains `WebAssembly.instantiate`, `WebAssembly.instantiateStreaming`, `WebAssembly.Module`, `WebAssembly.Memory`, `WebAssembly.Table` — WASM bytecode is inlined in the JS file (Emscripten `SINGLE_FILE=1` mode). No external `.wasm` dependency.
- **Build modules**: calib3d, core, dnn, features2d, flann, imgproc, js, objdetect, photo, video
- **Custom build**: Not required.

## Available APIs

### Working

| API | Notes |
|-----|-------|
| `cv.getPredefinedDictionary(dictId)` | Dict IDs on `cv` directly (e.g., `cv.DICT_4X4_50 = 0`), not `cv.aruco`. |
| `cv.aruco_CharucoBoard(size, squareLen, markerLen, dict, ids)` | 5 params. Pass `new cv.Mat()` for default IDs. |
| `board.generateImage(outSize, outMat, margin, borderBits)` | |
| `board.setLegacyPattern(bool)` / `board.getLegacyPattern()` | |
| `cv.aruco_CharucoParameters()` | Zero-arg. |
| `cv.aruco_DetectorParameters()` | Zero-arg. |
| `cv.aruco_RefineParameters(minRepDist, errCorrRate, checkAll)` | 3 params required. Defaults: `(10, 3, true)`. |
| `cv.aruco_CharucoDetector(board, charucoParams, detectorParams, refineParams)` | 4 params required. |
| `detector.detectBoard(img, corners, ids, markerCorners, markerIds)` | Corners: `Nx1 CV_32FC2`. IDs: `Nx1 CV_32SC1`. |
| `cv.calibrateCameraExtended(...)` | Full extended signature. |
| `cv.solvePnP`, `cv.Rodrigues`, `cv.projectPoints` | |
| `cv.cvtColor`, `cv.resize`, `cv.GaussianBlur`, `cv.goodFeaturesToTrack` | |

### Broken Bindings (with workarounds)

| API | Issue | Workaround |
|-----|-------|------------|
| `board.matchImagePoints(...)` | Unbound `vector<Point3f>` type. | **Manual computation** from board geometry: corner ID `i` → `((i % innerX + 1) * squareLen, (floor(i / innerX) + 1) * squareLen, 0)`. Verified against Python `board.getChessboardCorners()` for both standard and legacy patterns. |
| `board.getChessboardCorners()` | Same unbound type. | Same as above. |
| `cv.cornerSubPix(...)` | Not exposed. | `CharucoDetector` performs internal sub-pixel refinement (`CORNER_REFINE_SUBPIX` default). Single-pass `detectBoard` at full resolution gives sub-pixel accuracy. Verified: corner coordinates match Python ≤ 1e-9 px. |
| `cv.calibrateCamera(...)` | Not exposed. | Use `cv.calibrateCameraExtended(...)`. |
| `cv.aruco` namespace | Does not exist. | Dictionary constants are on `cv` directly. |

## Verified 2026-10-26

Full pipeline tested via Playwright headless Chromium, using REAL `src/core/detection.ts` and `src/core/calibration.ts` code (confirmed by `CORE_VERSION` assertion in test harness).
