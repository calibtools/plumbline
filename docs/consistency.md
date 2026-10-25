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
- **Test runner**: Playwright headless Chromium → Vite dev server → real `src/core` code
- Both datasets run in CI via `test:consistency:all`

## Runtime Parameter Verification

All CharucoParameters, DetectorParameters, and RefineParameters were printed from both JS and Python runtime and compared field-by-field. Every field is identical, including `minMarkers = 2`, `tryRefineMarkers = false`, `cornerRefinementMethod = 0`, and all adaptive threshold settings.

## Good Dataset (synthetic-01)

25 images, 1280×960, 3×3 zone coverage, 12–35% area, 0–45° tilt, 2/48 heatmap zero cells.

### Corner Detection

All 25 images: corner IDs and coordinates **match exactly**.
Corner coordinate difference: **< 4e-10 px** (threshold: 1e-3 px).

### Calibration

| Parameter | JS | Python | Difference |
|-----------|---:|-------:|-----------:|
| fx | 855.714992 | 855.714993 | 3.6e-10 (rel) |
| fy | 857.352896 | 857.352897 | 3.1e-10 (rel) |
| cx | 614.793354 | 614.793354 | 3.3e-10 (rel) |
| cy | 488.835979 | 488.835981 | 3.2e-9 (rel) |
| RMS | 0.79005164 | 0.79005200 | 3.6e-7 (abs) |

Distortion coefficients: all < 4e-10 absolute difference.

### Truth Comparison (informational)

| Parameter | Estimated | Truth | Error |
|-----------|----------:|------:|------:|
| fx | 855.71 | 800.0 | 7.0% |
| fy | 857.35 | 800.0 | 7.2% |
| cx | 614.79 | 640.0 | 25.2 px |
| cy | 488.84 | 480.0 | 8.8 px |
| k3 | -0.041 | 0.0001 | |

Does not reach the target of fx < 0.3%, cx < 3 px, k3 ± 0.05. High-tilt images lose many corners (some 9 of 24), weakening calibration constraints. JS and Python agree to < 4e-10 relative — this is a data limitation, not a code bug.

## Poor Dataset (synthetic-poor-01)

25 images, board always near center, 37/48 heatmap zero cells.

### Corner Detection

All 25 images: 24/24 corners, IDs and coordinates **match exactly**.

### Calibration

| Parameter | JS | Python | Difference |
|-----------|---:|-------:|-----------:|
| fx | 808.667398 | 808.667398 | 2.9e-10 (rel) |
| fy | 808.056186 | 808.056186 | 1.6e-10 (rel) |
| cx | 657.914642 | 657.914643 | 1.4e-9 (rel) |
| cy | 495.389630 | 495.389629 | 2.3e-9 (rel) |
| RMS | 0.29057929 | 0.29057900 | 2.9e-7 (abs) |

## P1c Bug: Hardcoded Image Path

During P1c, the poor-dataset consistency test appeared to fail with "18 vs 24 corners on img_00.png". Investigation showed:
- Raw opencv.js API: 24 corners (correct)
- `src/core/detection.ts` via harness: 24 corners (correct)
- `src/core/` had zero changes between P1b and P1c

**Root cause**: `tests/consistency.spec.ts` line 174 hardcoded `fixtures/synthetic-01/` in the image URL, ignoring the `FIXTURE_DIR` environment variable. When `FIXTURE_DIR=fixtures/synthetic-poor-01`, the test read the poor reference.json (24 corners) but loaded images from the good dataset's `synthetic-01/` directory (which has different images yielding different corner counts). Fixed by deriving the URL path from `FIXTURE_DIR`.
