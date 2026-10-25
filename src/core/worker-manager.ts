import type {
  BoardConfig,
  CalibrationFlags,
  CalibrationResult,
  ImageDetection,
  WorkerRequest,
  WorkerResponse,
} from './types';

// Vite resolves this URL at build time; the worker file is emitted as a separate chunk.
const workerUrl = new URL('./cv-worker.ts', import.meta.url);

type ProgressCallback = (message: string, percent: number) => void;

function createWorker(): Worker {
  return new Worker(workerUrl, { type: 'module' });
}

/**
 * Run detection on a single image in the web worker.
 * imageBuffer is transferred (zero-copy) to the worker.
 */
export function detectInWorker(
  imageBuffer: ArrayBuffer,
  filename: string,
  width: number,
  height: number,
  boardConfig: BoardConfig,
  onProgress?: ProgressCallback,
): Promise<ImageDetection> {
  return new Promise((resolve, reject) => {
    const worker = createWorker();

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'detect-result') {
        worker.terminate();
        resolve(msg.detection);
      } else if (msg.type === 'progress') {
        onProgress?.(msg.message, msg.percent);
      } else if (msg.type === 'error') {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };

    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message || 'Worker error'));
    };

    const request: WorkerRequest = {
      type: 'detect',
      imageData: imageBuffer,
      filename,
      width,
      height,
      boardConfig,
    };

    // Transfer the ArrayBuffer (zero-copy)
    worker.postMessage(request, [imageBuffer]);
  });
}

/**
 * Run calibration in the web worker.
 */
export function calibrateInWorker(
  detections: ImageDetection[],
  boardConfig: BoardConfig,
  flags: CalibrationFlags,
  imageSize: { width: number; height: number },
  onProgress?: ProgressCallback,
): Promise<CalibrationResult> {
  return new Promise((resolve, reject) => {
    const worker = createWorker();

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'calibrate-result') {
        worker.terminate();
        resolve(msg.result);
      } else if (msg.type === 'progress') {
        onProgress?.(msg.message, msg.percent);
      } else if (msg.type === 'error') {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };

    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message || 'Worker error'));
    };

    const request: WorkerRequest = {
      type: 'calibrate',
      detections,
      boardConfig,
      flags,
      imageSize,
    };

    worker.postMessage(request);
  });
}
