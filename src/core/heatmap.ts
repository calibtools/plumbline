import type { DetectedCorner, HeatmapCell, ImageDetection } from './types';

/**
 * Build a corner coverage heatmap by dividing the image into a grid
 * and counting how many detected corners fall into each cell.
 */
export function buildHeatmap(
  detections: ImageDetection[],
  imageWidth: number,
  imageHeight: number,
  gridCols: number,
  gridRows: number,
): HeatmapCell[] {
  const cellW = imageWidth / gridCols;
  const cellH = imageHeight / gridRows;

  // Initialize grid
  const counts: number[][] = [];
  for (let r = 0; r < gridRows; r++) {
    counts.push(new Array(gridCols).fill(0));
  }

  // Count corners from included images
  for (const det of detections) {
    if (!det.included) continue;
    for (const corner of det.corners) {
      const col = Math.min(Math.floor(corner.x / cellW), gridCols - 1);
      const row = Math.min(Math.floor(corner.y / cellH), gridRows - 1);
      counts[row][col]++;
    }
  }

  // Build cell array
  const cells: HeatmapCell[] = [];
  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      cells.push({ row: r, col: c, count: counts[r][c] });
    }
  }

  return cells;
}

/**
 * Aggregate cell counts into a 3×3 region grid (top/middle/bottom × left/center/right).
 * Returns a Map from region name to total corner count.
 */
export function aggregateRegions(
  cells: HeatmapCell[],
  gridCols: number,
  gridRows: number,
): Map<string, number> {
  const regions = new Map<string, number>();
  const names = ['top-left', 'top-center', 'top-right',
                 'middle-left', 'middle-center', 'middle-right',
                 'bottom-left', 'bottom-center', 'bottom-right'];
  for (const n of names) regions.set(n, 0);

  for (const cell of cells) {
    const vIdx = cell.row < gridRows / 3 ? 0 : cell.row >= (gridRows * 2) / 3 ? 2 : 1;
    const hIdx = cell.col < gridCols / 3 ? 0 : cell.col >= (gridCols * 2) / 3 ? 2 : 1;
    const name = names[vIdx * 3 + hIdx];
    regions.set(name, (regions.get(name) || 0) + cell.count);
  }

  return regions;
}

/**
 * Generate coverage advice based on the heatmap.
 *
 * Aggregates cell counts into 3×3 regions, then reports regions whose total
 * is below 20% of the mean region count as "sparse".
 */
export function getCoverageAdvice(
  cells: HeatmapCell[],
  gridCols: number,
  gridRows: number,
): string | null {
  const totalCorners = cells.reduce((s, c) => s + c.count, 0);
  if (totalCorners === 0) return null;

  const regions = aggregateRegions(cells, gridCols, gridRows);
  const regionCounts = [...regions.values()];
  const meanRegion = regionCounts.reduce((a, b) => a + b, 0) / regionCounts.length;
  const threshold = meanRegion * 0.2; // sparse = below 20% of mean

  const sparse: string[] = [];
  for (const [name, count] of regions) {
    if (count < threshold) {
      sparse.push(name);
    }
  }

  if (sparse.length === 0) return null;

  if (sparse.length >= 7) {
    return 'Coverage is sparse overall. Add more images with the board in different positions.';
  }

  return `Sparse coverage in: ${sparse.join(', ')}. Place the board in those areas for a few more shots.`;
}

/**
 * Analyze tilt distribution and return advice.
 */
export function getTiltAdvice(tilts: number[]): string | null {
  if (tilts.length === 0) return null;
  const frontalCount = tilts.filter((t) => t < 15).length;
  const frontalRatio = frontalCount / tilts.length;

  if (frontalRatio >= 0.8) {
    return 'Most images are nearly frontal (< 15°). Add a few shots at 30–45° tilt for better calibration.';
  }
  return null;
}

/**
 * Group corners from all included detections by their pixel position.
 * Used for rendering the heatmap overlay.
 */
export function getAllIncludedCorners(detections: ImageDetection[]): DetectedCorner[] {
  const corners: DetectedCorner[] = [];
  for (const det of detections) {
    if (!det.included) continue;
    corners.push(...det.corners);
  }
  return corners;
}
