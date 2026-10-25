# Detection Pipeline

## Two-Pass Detection

For images larger than 2000 px on the long edge, Plumbline uses a two-pass approach:

1. **Pass 1 (reduced resolution)**: The image is downscaled so the long edge is ≤ 2000 px. `CharucoDetector.detectBoard()` runs on this reduced image to quickly find markers and charuco corners. This pass determines *which* corners are present and provides initial rough positions.

2. **Pass 2 (full resolution)**: If any corners were found in pass 1, `CharucoDetector.detectBoard()` runs again on the **original full-resolution image**. The corner coordinates from this pass are used as the final detection result.

### Why not `cornerSubPix` on original resolution?

The official opencv.js 4.9.0 build does not expose `cv.cornerSubPix`. However, `CharucoDetector` performs internal sub-pixel refinement as part of `detectBoard()` (controlled by `DetectorParameters.cornerRefinementMethod`, which defaults to `CORNER_REFINE_SUBPIX`). Running `detectBoard` at full resolution therefore gives sub-pixel-accurate corner positions without needing a separate `cornerSubPix` call.

This is verified by the consistency tests — corner coordinates match Python OpenCV's output to within 1e-3 px.

## Inclusion Criteria

A detection is **excluded** from calibration if:

- Fewer than 6 charuco corners were detected
- Fewer than 4 unique corner IDs (degenerate for PnP)
- All detected corners are collinear (same row or column of the charuco grid)
- User manually excluded the image

## Object Point Computation

The official opencv.js binding for `board.matchImagePoints()` and `board.getChessboardCorners()` is broken (unbound `vector<Point3f>` type). Plumbline computes object points manually:

For a board with `squaresX × squaresY` squares, the inner charuco corner grid is `(squaresX-1) × (squaresY-1)`. Corner ID `i` maps to:

```
col = i % (squaresX - 1)
row = floor(i / (squaresX - 1))
objectPoint = ((col + 1) * squareLength, (row + 1) * squareLength, 0)
```

This matches OpenCV's internal layout and is verified by the consistency tests.

## Image Format Support

- **JPG/JPEG**: Supported
- **PNG**: Supported
- **HEIC/HEIF**: Not supported (browser decoding not universally available). Users are prompted to convert to JPG/PNG.
