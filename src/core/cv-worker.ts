/**
 * Web Worker entry point for detection and calibration.
 * opencv.js is loaded via importScripts (classic worker in production)
 * or via sync XHR (module worker in Vite dev).
 */
/// <reference lib="webworker" />

// Load opencv.js into worker global scope
try {
  // Works in classic workers (production build)
  importScripts('/opencv/opencv-4.9.0.js');
} catch {
  // Module workers (Vite dev): importScripts unavailable
  // Use synchronous XHR to load and eval
  const xhr = new XMLHttpRequest();
  xhr.open('GET', '/opencv/opencv-4.9.0.js', false);
  xhr.send();
  if (xhr.status === 200) {
    (0, eval)(xhr.responseText);
  }
}

import type { WorkerRequest, WorkerResponse } from './types';
import { detectCharucoCorners, buildDetection } from './detection';
import { calibrate } from './calibration';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

function post(msg: WorkerResponse) {
  ctx.postMessage(msg);
}

// Wait for opencv WASM to finish initializing
function waitForCV(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof cv !== 'undefined' && typeof cv.Mat === 'function') {
      resolve();
      return;
    }
    // cv might be a promise (WASM async init)
    const g = self as unknown as Record<string, unknown>;
    const maybeCV = g.cv;
    if (maybeCV && typeof (maybeCV as Promise<unknown>).then === 'function') {
      (maybeCV as Promise<unknown>).then((mod) => {
        g.cv = mod;
        resolve();
      });
      return;
    }
    // Poll
    const check = () => {
      if (typeof cv !== 'undefined' && typeof cv.Mat === 'function') {
        resolve();
      } else {
        setTimeout(check, 10);
      }
    };
    check();
  });
}

let ready = false;

ctx.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  if (!ready) {
    await waitForCV();
    ready = true;
  }

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
