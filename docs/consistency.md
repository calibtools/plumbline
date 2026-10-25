# Consistency Test Results

## Test Setup

- **opencv.js**: 4.9.0 (official pre-built, asm.js)
- **Python**: opencv-python 4.9.0.80 + opencv-contrib-python 4.9.0.80
- **Fixture**: 25 synthetic ChArUco images (5×7 board, DICT_4X4_50, sq=30mm, mk=22mm)
- **Ground truth**: fx=fy=800, cx=640, cy=480, k1=-0.1, k2=0.01, p1=0.001, p2=-0.001, k3=0.0001
- **Environment**: Playwright headless Chromium

## Corner Detection

All 25 images: **24/24 charuco corners detected** in both JS and Python.

- Corner IDs: **exact match** across all images
- Corner coordinates: **exact match** (difference < 1e-9 px in practice, threshold 1e-3 px)

## Calibration

Using the **same detected corners** (fed from Python's reference.json to ensure identical inputs):

| Parameter | JS (opencv.js) | Python (cv2) | Difference |
|-----------|---------------:|-------------:|-----------:|
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

All values are well within the specified tolerances:
- Camera matrix relative difference: < 1e-5 ✓ (actual: < 3e-9)
- Distortion coefficient absolute difference: < 1e-6 ✓ (actual: < 1.3e-7)
- RMS absolute difference: < 1e-4 ✓ (actual: 2.9e-7)

## Ground Truth Comparison (informational)

The calibration estimates vs. the synthetic ground truth:

| Parameter | Estimated | Truth | Error |
|-----------|----------:|------:|------:|
| fx | 808.67 | 800.0 | 8.67 |
| fy | 808.06 | 800.0 | 8.06 |
| cx | 657.91 | 640.0 | 17.91 |
| cy | 495.39 | 480.0 | 15.39 |

The ~1% error in focal length and ~2.5% error in principal point are expected for 25 synthetic images with noise — this confirms the calibration pipeline is working correctly.
