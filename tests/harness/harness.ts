/**
 * Test harness: bundles the REAL src/core code and exposes it to Playwright.
 *
 * Two modes:
 * - window.plumbline.run()       — direct main-thread execution (for consistency tests)
 * - window.plumbline.runWorker() — web worker execution (for responsiveness tests)
 *
 * Both use the same src/core detection + calibration code.
 */
import { CORE_VERSION } from '../../src/core/types';
import type {
  BoardConfig,
  CalibrationFlags,
  CalibrationResult,
  ImageDetection,
} from '../../src/core/types';
import { detectCharucoCorners, buildDetection } from '../../src/core/detection';
import { calibrate } from '../../src/core/calibration';
import { detectInWorker, calibrateInWorker } from '../../src/core/worker-manager';

export interface HarnessRunResult {
  coreVersion: string;
  detections: Array<{
    filename: string;
    corners: Array<{ id: number; x: number; y: number }>;
    markerCount: number;
    included: boolean;
  }>;
  calibration: {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
    distCoeffs: number[];
    rms: number;
    perImageRms: number[];
  };
}

function waitForOpenCV(): Promise<void> {
  return new Promise((resolve) => {
    const check = () => {
      if (typeof cv !== 'undefined' && typeof cv.Mat === 'function') {
        resolve();
      } else {
        setTimeout(check, 50);
      }
    };
    check();
  });
}

async function loadImageAsRGBA(
  url: string,
): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const response = await fetch(url);
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  bitmap.close();
  return { data: imageData.data, width: canvas.width, height: canvas.height };
}

function buildResult(detections: ImageDetection[], result: CalibrationResult): HarnessRunResult {
  return {
    coreVersion: CORE_VERSION,
    detections: detections.map((d) => ({
      filename: d.filename,
      corners: d.corners,
      markerCount: d.markerCount,
      included: d.included,
    })),
    calibration: {
      fx: result.cameraMatrix.fx,
      fy: result.cameraMatrix.fy,
      cx: result.cameraMatrix.cx,
      cy: result.cameraMatrix.cy,
      distCoeffs: result.distCoeffs,
      rms: result.rms,
      perImageRms: result.perImage.map((p) => p.rms),
    },
  };
}

/**
 * Direct main-thread execution (for consistency tests).
 */
async function plumblineRun(
  imageUrls: string[],
  boardConfig: BoardConfig,
  flags: CalibrationFlags,
): Promise<HarnessRunResult> {
  await waitForOpenCV();

  const detections: ImageDetection[] = [];
  for (const url of imageUrls) {
    const { data, width, height } = await loadImageAsRGBA(url);
    const filename = url.split('/').pop() || url;
    const { corners, markerCount } = detectCharucoCorners(data, width, height, boardConfig);
    detections.push(buildDetection(filename, width, height, corners, markerCount, boardConfig));
  }

  const included = detections.filter((d) => d.included);
  const imageSize = included.length > 0
    ? { width: included[0].width, height: included[0].height }
    : { width: 0, height: 0 };

  const result = calibrate(detections, boardConfig, flags, imageSize);
  return buildResult(detections, result);
}

/**
 * Web worker execution (for responsiveness tests).
 * Detection runs in web worker; main thread stays free.
 */
async function plumblineRunWorker(
  imageUrls: string[],
  boardConfig: BoardConfig,
  flags: CalibrationFlags,
): Promise<HarnessRunResult> {
  const detections: ImageDetection[] = [];

  for (const url of imageUrls) {
    const { data, width, height } = await loadImageAsRGBA(url);
    const filename = url.split('/').pop() || url;
    // Transfer the buffer to the worker (zero-copy)
    const detection = await detectInWorker(
      data.buffer, filename, width, height, boardConfig,
    );
    detections.push(detection);
  }

  const included = detections.filter((d) => d.included);
  const imageSize = included.length > 0
    ? { width: included[0].width, height: included[0].height }
    : { width: 0, height: 0 };

  const result = await calibrateInWorker(detections, boardConfig, flags, imageSize);
  return buildResult(detections, result);
}

// Expose to Playwright
declare global {
  interface Window {
    plumbline: {
      run: typeof plumblineRun;
      runWorker: typeof plumblineRunWorker;
      CORE_VERSION: string;
      ready: boolean;
    };
  }
}

waitForOpenCV().then(() => {
  window.plumbline = {
    run: plumblineRun,
    runWorker: plumblineRunWorker,
    CORE_VERSION,
    ready: true,
  };
});
