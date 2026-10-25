/// <reference lib="webworker" />
import type { WorkerRequest, WorkerResponse } from './types';
import { buildDetection, detectCharucoCorners } from './detection';
import { calibrate } from './calibration';

declare const self: DedicatedWorkerGlobalScope;

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;

  try {
    if (msg.type === 'detect') {
      post({ type: 'progress', message: `Detecting: ${msg.filename}`, percent: -1 });

      const pixels = new Uint8ClampedArray(msg.imageData);
      const { corners, markerCount } = detectCharucoCorners(
        pixels,
        msg.width,
        msg.height,
        msg.boardConfig,
      );

      const detection = buildDetection(
        msg.filename,
        msg.width,
        msg.height,
        corners,
        markerCount,
        msg.boardConfig,
      );

      post({ type: 'detect-result', detection });
    } else if (msg.type === 'calibrate') {
      post({ type: 'progress', message: 'Calibrating...', percent: -1 });

      const result = calibrate(msg.detections, msg.boardConfig, msg.flags, msg.imageSize);

      post({ type: 'calibrate-result', result });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    post({ type: 'error', message });
  }
};
