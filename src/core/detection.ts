import type { BoardConfig, DetectedCorner, ImageDetection } from './types';

const MIN_CORNERS = 6;
const MAX_DETECT_EDGE = 2000;

/**
 * Compute object (3D) points for a CharucoBoard from corner IDs.
 * For a board with squaresX × squaresY squares, inner corners form a
 * (squaresX-1) × (squaresY-1) grid. Corner ID `i` maps to:
 *   col = i % (squaresX - 1), row = floor(i / (squaresX - 1))
 *   objPoint = ((col + 1) * squareLength, (row + 1) * squareLength, 0)
 */
export function cornerIdToObjectPoint(
  id: number,
  board: BoardConfig,
): { x: number; y: number; z: number } {
  const innerX = board.squaresX - 1;
  const col = id % innerX;
  const row = Math.floor(id / innerX);
  return {
    x: (col + 1) * board.squareLength,
    y: (row + 1) * board.squareLength,
    z: 0,
  };
}

/**
 * Detect ChArUco corners in a single image.
 * imageData: raw RGBA pixel buffer
 */
export function detectCharucoCorners(
  imageData: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  boardConfig: BoardConfig,
): { corners: DetectedCorner[]; markerCount: number } {
  // Create grayscale Mat from RGBA
  const rgbaMat = new cv.Mat(height, width, cv.CV_8UC4);
  rgbaMat.data.set(imageData);
  const grayFull = new cv.Mat();
  cv.cvtColor(rgbaMat, grayFull, cv.COLOR_RGBA2GRAY);
  rgbaMat.delete();

  // Determine if we need to scale down for detection
  const maxEdge = Math.max(width, height);
  const scale = maxEdge > MAX_DETECT_EDGE ? MAX_DETECT_EDGE / maxEdge : 1;

  let grayDetect: cv.Mat;
  if (scale < 1) {
    grayDetect = new cv.Mat();
    const detectW = Math.round(width * scale);
    const detectH = Math.round(height * scale);
    cv.resize(grayFull, grayDetect, new cv.Size(detectW, detectH));
  } else {
    grayDetect = grayFull.clone();
  }

  // Set up board and detector
  const dict = cv.getPredefinedDictionary(boardConfig.dictionaryId);
  const emptyIds = new cv.Mat();
  const board = new cv.aruco_CharucoBoard(
    new cv.Size(boardConfig.squaresX, boardConfig.squaresY),
    boardConfig.squareLength,
    boardConfig.markerLength,
    dict,
    emptyIds,
  );
  emptyIds.delete();

  if (boardConfig.legacyPattern) {
    board.setLegacyPattern(true);
  }

  const charucoParams = new cv.aruco_CharucoParameters();
  const detectorParams = new cv.aruco_DetectorParameters();
  const refineParams = new cv.aruco_RefineParameters(10, 3, true);
  const detector = new cv.aruco_CharucoDetector(board, charucoParams, detectorParams, refineParams);

  // First pass: detect at reduced resolution
  const charucoCorners = new cv.Mat();
  const charucoIds = new cv.Mat();
  const markerCorners = new cv.MatVector();
  const markerIds = new cv.Mat();

  detector.detectBoard(grayDetect, charucoCorners, charucoIds, markerCorners, markerIds);
  const markerCount = markerCorners.size();
  grayDetect.delete();

  let finalCorners: DetectedCorner[];

  if (charucoCorners.rows > 0 && scale < 1) {
    // Second pass: detect at full resolution for sub-pixel accuracy
    const fullCorners = new cv.Mat();
    const fullIds = new cv.Mat();
    const fullMarkerCorners = new cv.MatVector();
    const fullMarkerIds = new cv.Mat();

    detector.detectBoard(grayFull, fullCorners, fullIds, fullMarkerCorners, fullMarkerIds);

    finalCorners = extractCorners(fullCorners, fullIds);

    fullCorners.delete();
    fullIds.delete();
    fullMarkerCorners.delete();
    fullMarkerIds.delete();
  } else {
    // No scaling needed — coordinates are already at full resolution
    finalCorners = extractCorners(charucoCorners, charucoIds);
  }

  // Cleanup
  charucoCorners.delete();
  charucoIds.delete();
  markerCorners.delete();
  markerIds.delete();
  detector.delete();
  charucoParams.delete();
  detectorParams.delete();
  refineParams.delete();
  board.delete();
  grayFull.delete();

  return { corners: finalCorners, markerCount };
}

function extractCorners(cornersMat: cv.Mat, idsMat: cv.Mat): DetectedCorner[] {
  const corners: DetectedCorner[] = [];
  for (let i = 0; i < cornersMat.rows; i++) {
    corners.push({
      id: idsMat.intAt(i, 0),
      x: cornersMat.data32F[i * 2],
      y: cornersMat.data32F[i * 2 + 1],
    });
  }
  return corners;
}

/**
 * Check if a detection should be included in calibration.
 */
export function shouldInclude(
  detection: ImageDetection,
  boardConfig: BoardConfig,
): { included: boolean; reason?: string } {
  if (detection.corners.length < MIN_CORNERS) {
    return {
      included: false,
      reason: `Too few corners detected (${detection.corners.length}, minimum ${MIN_CORNERS} required).`,
    };
  }

  // Check unique IDs sufficient for PnP (need at least 4 non-collinear points)
  const uniqueIds = new Set(detection.corners.map((c) => c.id));
  if (uniqueIds.size < 4) {
    return {
      included: false,
      reason: `Insufficient unique corner IDs (${uniqueIds.size}) to solve PnP.`,
    };
  }

  // Check collinearity: all corners on same row or column would be degenerate
  const innerX = boardConfig.squaresX - 1;
  const rows = new Set<number>();
  const cols = new Set<number>();
  for (const c of detection.corners) {
    rows.add(Math.floor(c.id / innerX));
    cols.add(c.id % innerX);
  }
  if (rows.size < 2 || cols.size < 2) {
    return {
      included: false,
      reason: 'Detected corners are collinear — cannot solve calibration.',
    };
  }

  return { included: true };
}

/**
 * Build the detection result for an image.
 */
export function buildDetection(
  filename: string,
  width: number,
  height: number,
  corners: DetectedCorner[],
  markerCount: number,
  boardConfig: BoardConfig,
): ImageDetection {
  const det: ImageDetection = {
    filename,
    width,
    height,
    corners,
    markerCount,
    included: true,
  };

  const check = shouldInclude(det, boardConfig);
  det.included = check.included;
  det.excludeReason = check.reason;

  return det;
}
