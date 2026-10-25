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
 * Generate coverage advice based on the heatmap.
 */
export function getCoverageAdvice(
  cells: HeatmapCell[],
  gridCols: number,
  gridRows: number,
): string | null {
  const maxCount = Math.max(...cells.map((c) => c.count));
  if (maxCount === 0) return null;

  const threshold = maxCount * 0.1; // cells with < 10% of max are "sparse"
  const sparseRegions: string[] = [];

  const regionName = (row: number, col: number): string => {
    const vName = row < gridRows / 3 ? 'top' : row >= (gridRows * 2) / 3 ? 'bottom' : 'middle';
    const hName = col < gridCols / 3 ? 'left' : col >= (gridCols * 2) / 3 ? 'right' : 'center';
    return `${vName}-${hName}`;
  };

  for (const cell of cells) {
    if (cell.count <= threshold) {
      sparseRegions.push(regionName(cell.row, cell.col));
    }
  }

  if (sparseRegions.length === 0) return null;

  // Deduplicate regions
  const unique = [...new Set(sparseRegions)];
  if (unique.length > cells.length * 0.5) {
    return 'Coverage is sparse overall. Add more images with the board in different positions.';
  }

  return `Sparse coverage in: ${unique.join(', ')}. Place the board in those areas for a few more shots.`;
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
