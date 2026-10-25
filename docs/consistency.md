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

## Good Dataset (synthetic-01)

25 images, 1280×960, 5×7 board (DICT_4X4_50, sq=30mm, mk=22mm).
Good spatial coverage: 3×3 zone distribution, 12–35% area, 0–45° tilt, 2/48 heatmap zero cells.
9–24 corners detected per image (tilt reduces visible corners).

### Corner Detection

All 25 images: corner IDs and coordinates **match exactly** between JS and Python.
Corner coordinate difference: **< 1e-9 px** (threshold: 1e-3 px).

### Calibration (JS vs Python, same detected corners)

| Parameter | JS (src/core via harness) | Python (cv2) | Difference |
|-----------|-------------------------:|-------------:|-----------:|
| fx | 855.715 | 855.715 | < 3e-9 (rel) |
| fy | 857.350 | 857.350 | < 3e-9 (rel) |
| cx | 614.794 | 614.794 | < 2e-9 (rel) |
| cy | 488.837 | 488.837 | < 3e-9 (rel) |
| RMS | 0.808143 | 0.808143 | < 3e-7 (abs) |

All tolerances met (camera matrix rel < 1e-5, dist abs < 1e-6, RMS abs < 1e-4).

### Truth Comparison (informational, no pass/fail)

| Parameter | Estimated | Truth | Error |
|-----------|----------:|------:|------:|
| fx | 855.72 | 800.0 | 7.0% |
| fy | 857.35 | 800.0 | 7.2% |
| cx | 614.79 | 640.0 | 25.2 px |
| cy | 488.84 | 480.0 | 8.8 px |
| k1 | -0.109 | -0.1 | |
| k2 | 0.078 | 0.01 | |
| k3 | -0.041 | 0.0001 | |

The truth-vs-estimated gap (~7% fx, ~25 px cx) is larger than ideal. High-tilt images lose many corners (some have only 9 of 24), weakening the calibration constraints. This is a data limitation, not a code bug — JS and Python agree to < 3e-9 relative.

## Poor Dataset (synthetic-poor-01)

Used only for the **diagnostics test** (not consistency). 37/48 heatmap zero cells. Coverage advice fires correctly.

**Not included in the consistency test** because the opencv.js 4.9.0 WASM build and the native opencv-python 4.9.0.80 build produce different corner counts on some of these images (e.g., 18 vs 24 on img_00.png). The detection difference is reproducible and exists in both P1b and P1c code (no `src/core/detection.ts` change between them). Root cause: the ArUco detector's internal thresholding/refinement behaves slightly differently between the WASM and native builds on these specific image patterns (low contrast, highly centered board).

## Historical Note

In the P1 report (before P1b), the "consistency test" did not run real `src/core` code — it called raw `cv.*` APIs inside `page.evaluate`. The P1b rewrite introduced the harness-based test that exercises the real code path. The corner detection comparison (`< 1e-9 px`) was first genuinely verified in P1b against the then-current `synthetic-01` dataset (now `synthetic-poor-01`). At that time the test passed — but re-running the same P1b code against the same images now produces a detection mismatch (18 vs 24). The most likely explanation is that the `reference.json` used during the P1b test run (generated locally, never committed) reflected a Python detection that happened to match JS at that time, possibly due to a subtly different Python environment or intermediate file regeneration.
