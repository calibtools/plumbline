/**
 * Diagnostics test: verifies coverage heatmap and tilt advice behave correctly
 * on known-good and known-poor datasets.
 *
 * - synthetic-poor-01: coverage advice SHOULD fire (corners clustered in center)
 * - synthetic-01: coverage advice should NOT fire, tilt advice should NOT fire
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, type Browser, type Page } from 'playwright';
import { type ChildProcess, spawn } from 'child_process';
import { readFileSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { createServer, type Server } from 'http';
import { buildHeatmap, getCoverageAdvice, getTiltAdvice } from '../src/core/heatmap';
import type { ImageDetection } from '../src/core/types';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');

function startFixtureServer(): Promise<{ server: Server; port: number }> {
  return new Promise((resolve) => {
    const srv = createServer((req, res) => {
      const url = new URL(req!.url!, 'http://localhost');
      const filePath = join(PROJECT_ROOT, url.pathname);
      try {
        const data = readFileSync(filePath);
        res.writeHead(200, { 'Content-Type': 'image/png', 'Access-Control-Allow-Origin': '*' });
        res.end(data);
      } catch { res.writeHead(404); res.end('Not found'); }
    });
    srv.listen(0, '127.0.0.1', () => {
      resolve({ server: srv, port: (srv.address() as { port: number }).port });
    });
  });
}

function startHarness(projectRoot: string): Promise<{ process: ChildProcess; port: number }> {
  const port = 15674 + Math.floor(Math.random() * 1000);
  const proc = spawn('npx', ['vite', '--config', 'tests/harness/vite.config.ts', '--port', String(port), '--strictPort'], {
    cwd: projectRoot, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, VITE_CONFIG_NATIVE_IGNORE_WARNING: 'true' },
  });
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('Vite timeout')), 30000);
    let o = '';
    const d = (data: Buffer) => { o += data.toString(); if (o.includes('ready in')) { clearTimeout(t); setTimeout(() => resolve({ process: proc, port }), 500); } };
    proc.stdout!.on('data', d); proc.stderr!.on('data', d);
    proc.on('exit', (c) => { clearTimeout(t); reject(new Error(`exit ${c}`)); });
  });
}

function computeTiltsFromTruth(truthPath: string): number[] {
  const truth = JSON.parse(readFileSync(truthPath, 'utf-8'));
  return truth.images.map((img: { rvec: number[] }) => {
    const rv = img.rvec;
    const angle = Math.sqrt(rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2);
    if (angle < 1e-10) return 0;
    const k2 = rv[2] / angle;
    const c = Math.cos(angle);
    const r22 = c + k2 * k2 * (1 - c);
    return Math.acos(Math.abs(r22)) * 180 / Math.PI;
  });
}

describe('Diagnostics', () => {
  let browser: Browser;
  let page: Page;
  let harnessProc: ChildProcess;
  let harnessPort: number;
  let fixtureSrv: Server;
  let fixturePort: number;

  beforeAll(async () => {
    const fs = await startFixtureServer();
    fixtureSrv = fs.server; fixturePort = fs.port;
    const h = await startHarness(PROJECT_ROOT);
    harnessProc = h.process; harnessPort = h.port;

    browser = await chromium.launch({ headless: true });
    page = await browser.newPage();
    await page.goto(`http://localhost:${harnessPort}/`);
    await page.waitForFunction(() => window.plumbline && window.plumbline.ready, { timeout: 60000 });
  }, 120000);

  afterAll(async () => { await browser?.close(); harnessProc?.kill(); fixtureSrv?.close(); });

  async function runDataset(fixtureDir: string) {
    const absDir = join(PROJECT_ROOT, fixtureDir);
    const imageFiles = readdirSync(absDir).filter((f) => f.endsWith('.png')).sort();
    const urls = imageFiles.map((f) => `http://127.0.0.1:${fixturePort}/${fixtureDir}/${f}`);

    const boardConfig = { squaresX: 5, squaresY: 7, squareLength: 30, markerLength: 22, dictionaryId: 0, legacyPattern: false };
    const flags = { fixK3: false, fixTangential: false, rationalModel: false, thinPrism: false };

    const result = await page.evaluate(
      async ({ urls, board, fl }) => window.plumbline.run(urls, board, fl),
      { urls, board: boardConfig, fl: flags },
    );

    // Build ImageDetection[] for heatmap (Node-side, same src/core code)
    const detections: ImageDetection[] = result.detections.map((d: { filename: string; corners: Array<{ id: number; x: number; y: number }>; markerCount: number; included: boolean }) => ({
      filename: d.filename, width: 1280, height: 960,
      corners: d.corners, markerCount: d.markerCount, included: d.included,
    }));

    const cells = buildHeatmap(detections, 1280, 960, 8, 6);
    const coverageAdvice = getCoverageAdvice(cells, 8, 6);
    const zeroCells = cells.filter((c) => c.count === 0).length;

    return { detections, cells, coverageAdvice, zeroCells, calibration: result.calibration };
  }

  it('synthetic-poor-01: coverage advice fires', async () => {
    if (!existsSync(join(PROJECT_ROOT, 'fixtures', 'synthetic-poor-01'))) {
      console.log('synthetic-poor-01 not found, skipping');
      return;
    }

    const { coverageAdvice, zeroCells } = await runDataset('fixtures/synthetic-poor-01');

    console.log(`synthetic-poor-01: ${zeroCells}/48 zero cells`);
    console.log(`Coverage advice: ${coverageAdvice}`);

    expect(coverageAdvice, 'Coverage advice should fire for poor dataset').not.toBeNull();
  }, 120000);

  it('synthetic-01: coverage advice does NOT fire, tilt advice does NOT fire', async () => {
    if (!existsSync(join(PROJECT_ROOT, 'fixtures', 'synthetic-01'))) {
      console.log('synthetic-01 not found, skipping');
      return;
    }

    const { coverageAdvice, zeroCells, cells } = await runDataset('fixtures/synthetic-01');

    console.log(`synthetic-01: ${zeroCells}/48 zero cells`);
    console.log('Heatmap grid:');
    for (let r = 0; r < 6; r++) {
      const row = cells.filter((c) => c.row === r).sort((a, b) => a.col - b.col);
      console.log('  ' + row.map((c) => String(c.count).padStart(4)).join(''));
    }
    console.log(`Coverage advice: ${coverageAdvice || '(none)'}`);

    expect(coverageAdvice, 'Coverage advice should NOT fire for good dataset').toBeNull();
    expect(zeroCells, 'At most 4 zero cells').toBeLessThanOrEqual(4);

    // Tilt check
    const truthPath = join(PROJECT_ROOT, 'fixtures', 'synthetic-01', 'truth.json');
    if (existsSync(truthPath)) {
      const tilts = computeTiltsFromTruth(truthPath);
      const tiltAdvice = getTiltAdvice(tilts);
      console.log(`Tilt advice: ${tiltAdvice || '(none)'}`);
      console.log(`Tilts: ${tilts.map((t) => t.toFixed(1)).join(', ')}`);
      expect(tiltAdvice, 'Tilt advice should NOT fire for good dataset').toBeNull();
    }
  }, 120000);
});
