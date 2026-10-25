# Consistency Test Results

## What is tested

The consistency test loads a **test harness page** (`tests/harness/`) that bundles the **real `src/core/detection.ts` and `src/core/calibration.ts`** via Vite. The harness exposes `window.plumbline.run()` which:

1. Loads each fixture image via `fetch` + `createImageBitmap`
2. Calls `detectCharucoCorners()` from `src/core/detection.ts`
3. Calls `calibrate()` from `src/core/calibration.ts`

An anti-bypass assertion checks `CORE_VERSION` from `src/core/types.ts` to prove the real code is running.

The Python reference uses opencv-python 4.9.0.80 with the same board parameters and default calibration flags.

## Setup

- **opencv.js**: 4.9.0 (official, single-file WASM)
- **Python**: opencv-python 4.9.0.80 + opencv-contrib-python 4.9.0.80
- **Fixture**: 25 synthetic ChArUco images, 1280×960, 5×7 board, DICT_4X4_50, sq=30mm, mk=22mm
- **Test runner**: Playwright headless Chromium → Vite dev server → real `src/core` code

## Corner Detection

All 25 images: **24/24 charuco corners detected** in both JS and Python.

- Corner IDs: **exact match** across all 25 images
- Corner coordinates: **< 1e-9 px difference** (threshold: 1e-3 px)

## Calibration

| Parameter | JS (src/core via harness) | Python (cv2) | Difference |
|-----------|-------------------------:|-------------:|-----------:|
| fx | 808.667398 | 808.667398 | 2.9e-10 (rel) |
| fy | 808.056186 | 808.056186 | 1.6e-10 (rel) |
| cx | 657.914642 | 657.914643 | 1.4e-9 (rel) |
| cy | 495.389630 | 495.389629 | 2.3e-9 (rel) |
| k1 | -9.249306e-2 | -9.249306e-2 | 3.4e-9 (abs) |
| k2 | 4.609493e-1 | 4.609493e-1 | 3.9e-8 (abs) |
| p1 | 4.142639e-4 | 4.142635e-4 | 4.1e-10 (abs) |
| p2 | 9.762812e-4 | 9.762816e-4 | 3.0e-10 (abs) |
| k3 | -1.352119e+0 | -1.352119e+0 | 1.2e-7 (abs) |
| **RMS** | **0.29057929** | **0.29057900** | **2.9e-7 (abs)** |

All tolerances met:
- Camera matrix relative diff: < 1e-5 threshold → actual < 3e-9
- Distortion coeffs absolute diff: < 1e-6 threshold → actual < 1.3e-7
- RMS absolute diff: < 1e-4 threshold → actual 2.9e-7

## Ground Truth Comparison (informational)

| Parameter | Estimated | Truth | Error |
|-----------|----------:|------:|------:|
| fx | 808.67 | 800.0 | 8.67 |
| fy | 808.06 | 800.0 | 8.06 |
| cx | 657.91 | 640.0 | 17.91 |
| cy | 495.39 | 480.0 | 15.39 |

~1% focal length error, ~2.5% principal point error — expected for 25 noisy synthetic images.
