/** Board configuration parameters */
export interface BoardConfig {
  squaresX: number;
  squaresY: number;
  squareLength: number;
  markerLength: number;
  dictionaryId: number;
  legacyPattern: boolean;
}

/** A single detected corner */
export interface DetectedCorner {
  id: number;
  x: number;
  y: number;
}

/** Detection result for a single image */
export interface ImageDetection {
  filename: string;
  width: number;
  height: number;
  corners: DetectedCorner[];
  markerCount: number;
  included: boolean;
  excludeReason?: string;
}

/** Calibration flags */
export interface CalibrationFlags {
  fixK3: boolean;
  fixTangential: boolean;
  rationalModel: boolean;
  thinPrism: boolean;
}

/** Per-image calibration result */
export interface PerImageResult {
  filename: string;
  rms: number;
  rvec: [number, number, number];
  tvec: [number, number, number];
  included: boolean;
}

/** Full calibration result */
export interface CalibrationResult {
  cameraMatrix: {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
  };
  distCoeffs: number[];
  rms: number;
  imageSize: { width: number; height: number };
  perImage: PerImageResult[];
  flags: CalibrationFlags;
  boardConfig: BoardConfig;
  version: string;
}

/** Coverage heatmap cell */
export interface HeatmapCell {
  row: number;
  col: number;
  count: number;
}

/** Worker message types */
export type WorkerRequest =
  | {
      type: 'detect';
      imageData: ArrayBuffer;
      filename: string;
      width: number;
      height: number;
      boardConfig: BoardConfig;
    }
  | {
      type: 'calibrate';
      detections: ImageDetection[];
      boardConfig: BoardConfig;
      flags: CalibrationFlags;
      imageSize: { width: number; height: number };
    };

export type WorkerResponse =
  | { type: 'detect-result'; detection: ImageDetection }
  | { type: 'calibrate-result'; result: CalibrationResult }
  | { type: 'progress'; message: string; percent: number }
  | { type: 'error'; message: string };
