/**
 * Consistency tests: compare browser-side ChArUco detection and calibration
 * (running the REAL src/core code via the test harness) against Python OpenCV
 * reference results.
 *
 * The harness page (tests/harness/) bundles src/core/detection.ts and
 * src/core/calibration.ts via Vite and exposes window.plumbline.run().
 * This test starts a Vite dev server for the harness, loads it in Playwright,
 * and calls the real pipeline.
 *
 * Run: pnpm test:consistency
 * Requires: fixtures/synthetic-01/reference.json
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, type Browser, type Page } from 'playwright';
import { type ChildProcess, spawn } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { createServer, type Server } from 'http';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');
const FIXTURE_DIR = process.env.FIXTURE_DIR || join(PROJECT_ROOT, 'fixtures', 'synthetic-01');
const REFERENCE_PATH = join(FIXTURE_DIR, 'reference.json');
const TRUTH_PATH = join(FIXTURE_DIR, 'truth.json');

const DICT_NAME_TO_ID: Record<string, number> = {
  DICT_4X4_50: 0, DICT_4X4_100: 1, DICT_4X4_250: 2, DICT_4X4_1000: 3,
  DICT_5X5_50: 4, DICT_5X5_100: 5, DICT_5X5_250: 6, DICT_5X5_1000: 7,
  DICT_6X6_50: 8, DICT_6X6_100: 9, DICT_6X6_250: 10, DICT_6X6_1000: 11,
  DICT_7X7_50: 12, DICT_7X7_100: 13, DICT_7X7_250: 14, DICT_7X7_1000: 15,
  DICT_ARUCO_ORIGINAL: 16,
};

interface ReferenceImage {
  filename: string;
  charuco_ids: number[];
  charuco_corners: Array<[number, number]>;
  num_markers: number;
}

interface ReferenceData {
  board: {
    squares_x: number;
    squares_y: number;
    square_length: number;
    marker_length: number;
    dictionary: string;
  };
  image_size: [number, number];
  images: ReferenceImage[];
  camera_matrix: number[];
  dist_coeffs: number[];
  rms: number;
  per_image_rms: number[];
}

interface TruthData {
  camera_matrix: number[];
  dist_coeffs: number[];
}

describe('Consistency with Python OpenCV', () => {
  let browser: Browser;
  let page: Page;
  let harnessProcess: ChildProcess;
  let fixtureServer: Server;
  let fixturePort: number;
  let harnessPort: number;
  let reference: ReferenceData;
  let truth: TruthData | null = null;

  beforeAll(async () => {
    if (!existsSync(REFERENCE_PATH)) {
      throw new Error(
        `Reference data not found at ${REFERENCE_PATH}. Run: pnpm generate:fixtures`,
      );
    }

    reference = JSON.parse(readFileSync(REFERENCE_PATH, 'utf-8'));
    if (existsSync(TRUTH_PATH)) {
      truth = JSON.parse(readFileSync(TRUTH_PATH, 'utf-8'));
    }

    // 1. Start a static file server for fixture images
    fixtureServer = createServer((req, res) => {
      const url = new URL(req!.url!, 'http://localhost');
      const filePath = join(PROJECT_ROOT, url.pathname);
      try {
        const data = readFileSync(filePath);
        const ext = filePath.split('.').pop();
        const mime: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', json: 'application/json' };
        res.writeHead(200, {
          'Content-Type': mime[ext || ''] || 'application/octet-stream',
          'Access-Control-Allow-Origin': '*',
        });
        res.end(data);
      } catch {
        res.writeHead(404);
        res.end('Not found');
      }
    });
    await new Promise<void>((resolve) => {
      fixtureServer.listen(0, '127.0.0.1', () => {
        const addr = fixtureServer.address();
        fixturePort = typeof addr === 'object' && addr ? addr.port : 0;
        resolve();
      });
    });

    // 2. Start Vite dev server for the harness
    harnessPort = 15174 + Math.floor(Math.random() * 1000);
    harnessProcess = spawn('npx', ['vite', '--config', 'tests/harness/vite.config.ts', '--port', String(harnessPort), '--strictPort'], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, VITE_CONFIG_NATIVE_IGNORE_WARNING: 'true' },
    });

    // Wait for Vite to be ready
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Vite harness did not start in 30s')), 30000);
      let output = '';
      const onData = (data: Buffer) => {
        output += data.toString();
        if (output.includes('ready in') || output.includes('Local:') || output.includes('localhost:')) {
          clearTimeout(timeout);
          // Give Vite a moment to fully bind
          setTimeout(resolve, 500);
        }
      };
      harnessProcess.stdout!.on('data', onData);
      harnessProcess.stderr!.on('data', onData);
      harnessProcess.on('exit', (code) => {
        clearTimeout(timeout);
        reject(new Error(`Vite exited with code ${code}. Output: ${output}`));
      });
    });

    // 3. Launch browser, load harness, wait for opencv + plumbline.ready
    browser = await chromium.launch({ headless: true });
    page = await browser.newPage();
    await page.goto(`http://localhost:${harnessPort}/`);
    await page.waitForFunction(
      () => window.plumbline && window.plumbline.ready === true,
      { timeout: 60000 },
    );
  }, 120000);

  afterAll(async () => {
    await browser?.close();
    harnessProcess?.kill();
    fixtureServer?.close();
  });

  it('harness loads real src/core code (CORE_VERSION exists)', async () => {
    const version = await page.evaluate(() => window.plumbline.CORE_VERSION);
    expect(version).toBeTruthy();
    expect(typeof version).toBe('string');
    console.log(`CORE_VERSION: ${version}`);
  });

  it('detects the same corners and produces matching calibration', async () => {
    const boardConfig = {
      squaresX: reference.board.squares_x,
      squaresY: reference.board.squares_y,
      squareLength: reference.board.square_length,
      markerLength: reference.board.marker_length,
      dictionaryId: DICT_NAME_TO_ID[reference.board.dictionary] ?? 0,
      legacyPattern: false,
    };

    const imageUrls = reference.images.map(
      (img) => {
        // FIXTURE_DIR may be absolute or relative; make a path relative to PROJECT_ROOT for the URL
        const relDir = FIXTURE_DIR.startsWith(PROJECT_ROOT)
          ? FIXTURE_DIR.slice(PROJECT_ROOT.length + 1)
          : FIXTURE_DIR;
        return `http://127.0.0.1:${fixturePort}/${relDir}/${img.filename}`;
      },
    );

    const flags = { fixK3: false, fixTangential: false, rationalModel: false, thinPrism: false };

    // Call the REAL src/core pipeline
    const result = await page.evaluate(
      async ({ urls, board, fl }) => {
        return window.plumbline.run(urls, board, fl);
      },
      { urls: imageUrls, board: boardConfig, fl: flags },
    );

    // --- Verify CORE_VERSION came back (anti-bypass) ---
    expect(result.coreVersion).toBeTruthy();

    // --- Corner detection: count, IDs, coordinates ---
    for (let imgIdx = 0; imgIdx < reference.images.length; imgIdx++) {
      const refImg = reference.images[imgIdx];
      const jsImg = result.detections[imgIdx];

      const jsIds = jsImg.corners.map((c: { id: number }) => c.id);
      const jsCorners = jsImg.corners.map((c: { x: number; y: number }) => [c.x, c.y]);

      expect(
        jsIds.length,
        `Corner count mismatch for ${refImg.filename}: JS=${jsIds.length}, Py=${refImg.charuco_ids.length}`,
      ).toBe(refImg.charuco_ids.length);

      expect(
        jsIds,
        `Corner IDs mismatch for ${refImg.filename}`,
      ).toEqual(refImg.charuco_ids);

      for (let i = 0; i < jsCorners.length; i++) {
        const dx = Math.abs(jsCorners[i][0] - refImg.charuco_corners[i][0]);
        const dy = Math.abs(jsCorners[i][1] - refImg.charuco_corners[i][1]);
        expect(dx, `Corner ${refImg.charuco_ids[i]} X diff for ${refImg.filename}: ${dx}`).toBeLessThan(1e-3);
        expect(dy, `Corner ${refImg.charuco_ids[i]} Y diff for ${refImg.filename}: ${dy}`).toBeLessThan(1e-3);
      }
    }

    // --- Calibration: camera matrix ---
    const refCM = reference.camera_matrix;
    const refFx = refCM[0], refFy = refCM[4], refCx = refCM[2], refCy = refCM[5];
    const cal = result.calibration;

    console.log(`\nCalibration comparison (via real src/core):`);
    console.log(`  fx: JS=${cal.fx.toFixed(6)}, Py=${refFx.toFixed(6)}, relDiff=${(Math.abs(cal.fx - refFx) / refFx).toExponential(3)}`);
    console.log(`  fy: JS=${cal.fy.toFixed(6)}, Py=${refFy.toFixed(6)}, relDiff=${(Math.abs(cal.fy - refFy) / refFy).toExponential(3)}`);
    console.log(`  cx: JS=${cal.cx.toFixed(6)}, Py=${refCx.toFixed(6)}, relDiff=${(Math.abs(cal.cx - refCx) / refCx).toExponential(3)}`);
    console.log(`  cy: JS=${cal.cy.toFixed(6)}, Py=${refCy.toFixed(6)}, relDiff=${(Math.abs(cal.cy - refCy) / refCy).toExponential(3)}`);
    console.log(`  RMS: JS=${cal.rms.toFixed(8)}, Py=${reference.rms.toFixed(8)}, diff=${Math.abs(cal.rms - reference.rms).toExponential(3)}`);

    expect(Math.abs(cal.fx - refFx) / refFx, 'fx relative diff').toBeLessThan(1e-5);
    expect(Math.abs(cal.fy - refFy) / refFy, 'fy relative diff').toBeLessThan(1e-5);
    expect(Math.abs(cal.cx - refCx) / refCx, 'cx relative diff').toBeLessThan(1e-5);
    expect(Math.abs(cal.cy - refCy) / refCy, 'cy relative diff').toBeLessThan(1e-5);

    // --- Distortion coefficients ---
    for (let i = 0; i < Math.min(cal.distCoeffs.length, reference.dist_coeffs.length); i++) {
      const diff = Math.abs(cal.distCoeffs[i] - reference.dist_coeffs[i]);
      console.log(`  dist[${i}]: JS=${cal.distCoeffs[i].toExponential(6)}, Py=${reference.dist_coeffs[i].toExponential(6)}, diff=${diff.toExponential(3)}`);
      expect(diff, `dist[${i}] absolute diff`).toBeLessThan(1e-6);
    }

    // --- RMS ---
    expect(
      Math.abs(cal.rms - reference.rms),
      `RMS diff: JS=${cal.rms}, Py=${reference.rms}`,
    ).toBeLessThan(1e-4);

    // --- Ground truth comparison (informational) ---
    if (truth) {
      console.log(`\nComparison with ground truth (informational):`);
      console.log(`  fx: ${cal.fx.toFixed(2)} vs truth ${truth.camera_matrix[0]}, error=${Math.abs(cal.fx - truth.camera_matrix[0]).toFixed(4)}`);
      console.log(`  fy: ${cal.fy.toFixed(2)} vs truth ${truth.camera_matrix[4]}, error=${Math.abs(cal.fy - truth.camera_matrix[4]).toFixed(4)}`);
      console.log(`  cx: ${cal.cx.toFixed(2)} vs truth ${truth.camera_matrix[2]}, error=${Math.abs(cal.cx - truth.camera_matrix[2]).toFixed(4)}`);
      console.log(`  cy: ${cal.cy.toFixed(2)} vs truth ${truth.camera_matrix[5]}, error=${Math.abs(cal.cy - truth.camera_matrix[5]).toFixed(4)}`);
      for (let i = 0; i < Math.min(cal.distCoeffs.length, truth.dist_coeffs.length); i++) {
        console.log(`  dist[${i}]: ${cal.distCoeffs[i].toExponential(4)} vs truth ${truth.dist_coeffs[i].toExponential(4)}`);
      }
    }
  }, 300000);
});
