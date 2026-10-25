# Detection Pipeline

## Single-Pass Detection

Plumbline runs `CharucoDetector.detectBoard()` once on the **original full-resolution image**. The detector performs internal sub-pixel refinement (`CORNER_REFINE_SUBPIX` is the default `cornerRefinementMethod`).

This produces corner coordinates that match Python OpenCV to within 1e-9 px (see `docs/consistency.md`).

## Why not cornerSubPix?

`cv.cornerSubPix` is not exposed in the official opencv.js 4.9.0 build. Since `CharucoDetector.detectBoard()` already applies sub-pixel refinement internally, a separate `cornerSubPix` call is unnecessary for ChArUco workflows.

## 12 MP (4000×3000) Detection Timing

Measured on Apple M-series (headless Chromium, single-file WASM opencv.js 4.9.0):

| Image | Time |
|-------|-----:|
| img_00.png | 308 ms |
| img_01.png | 227 ms |
| img_02.png | 237 ms |
| img_03.png | 209 ms |
| img_04.png | 233 ms |
| **Average** | **243 ms** |
| **Max** | **308 ms** |

Detection runs in a Web Worker; the main thread stays responsive (max rAF gap 19.9 ms during 25-image batch).

## Object Point Computation

`board.matchImagePoints()` and `board.getChessboardCorners()` are broken in opencv.js (unbound `vector<Point3f>` type). Plumbline computes object points manually:

For a board with `squaresX × squaresY` squares, inner corners form a `(squaresX-1) × (squaresY-1)` grid. Corner ID `i`:

```
col = i % (squaresX - 1)
row = floor(i / (squaresX - 1))
objectPoint = ((col + 1) * squareLength, (row + 1) * squareLength, 0)
```

This formula works for both standard and legacy patterns — verified against Python `board.getChessboardCorners()` output for both modes (see `fixtures/board-geometry/`).

## Coverage Heatmap Sparse Region Definition

The heatmap divides the image into an N×M grid (default 8×6). For coverage advice, individual cell counts are aggregated into a 3×3 region grid (top/middle/bottom × left/center/right). A region is considered **sparse** if its total corner count is below **20% of the mean region count**.

This prevents the bug where individual zero-cells in a high-count region (e.g., "middle-center" having 235 corners overall but some individual cells at zero) are falsely flagged. The aggregation ensures regions with good overall coverage are not reported as sparse.

If 7+ of 9 regions are sparse, the advice says "coverage is sparse overall" instead of listing individual regions.

## Inclusion Criteria

A detection is excluded from calibration if:
- Fewer than 6 charuco corners detected
- Fewer than 4 unique corner IDs (degenerate for PnP)
- All detected corners are collinear
- User manually excluded the image

## Image Format Support

- **JPG/JPEG**: Supported
- **PNG**: Supported
- **HEIC/HEIF**: Not supported (prompted to convert)
