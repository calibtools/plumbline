/**
 * Worker responsiveness test: verifies the main thread stays responsive
 * while the web worker processes images.
 *
 * Loads the test harness, starts a rAF counter on the main thread,
 * processes 25 synthetic images via the real worker pipeline, and checks
 * that no two adjacent rAF frames are more than 100ms apart.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, type Browser, type Page } from 'playwright';
import { type ChildProcess, spawn } from 'child_process';
import { readFileSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { createServer, type Server } from 'http';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');
const FIXTURE_DIR = join(PROJECT_ROOT, 'fixtures', 'synthetic-01');

describe('Worker responsiveness', () => {
  let browser: Browser;
  let page: Page;
  let harnessProcess: ChildProcess;
  let fixtureServer: Server;
  let fixturePort: number;
  let harnessPort: number;

  beforeAll(async () => {
    if (!existsSync(FIXTURE_DIR)) {
      throw new Error(`Fixtures not found at ${FIXTURE_DIR}. Run: pnpm generate:fixtures`);
    }

    // Static file server for fixtures
    fixtureServer = createServer((req, res) => {
      const url = new URL(req!.url!, 'http://localhost');
      const filePath = join(PROJECT_ROOT, url.pathname);
      try {
        const data = readFileSync(filePath);
        const ext = filePath.split('.').pop();
        const mime: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg' };
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
        fixturePort = (fixtureServer.address() as { port: number }).port;
        resolve();
      });
    });

    // Vite harness dev server
    harnessPort = 15274 + Math.floor(Math.random() * 1000);
    harnessProcess = spawn('npx', ['vite', '--config', 'tests/harness/vite.config.ts', '--port', String(harnessPort), '--strictPort'], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, VITE_CONFIG_NATIVE_IGNORE_WARNING: 'true' },
    });

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Vite harness did not start')), 30000);
      let output = '';
      const onData = (data: Buffer) => {
        output += data.toString();
        if (output.includes('ready in') || output.includes('Local:')) {
          clearTimeout(timeout);
          setTimeout(resolve, 500);
        }
      };
      harnessProcess.stdout!.on('data', onData);
      harnessProcess.stderr!.on('data', onData);
      harnessProcess.on('exit', (code) => {
        clearTimeout(timeout);
        reject(new Error(`Vite exited ${code}. ${output}`));
      });
    });

    browser = await chromium.launch({ headless: true });
    page = await browser.newPage();
    page.on('console', (msg) => {
      if (msg.type() === 'error') console.log('[BROWSER ERROR]', msg.text());
    });
    page.on('pageerror', (err) => console.log('[PAGE ERROR]', err.message));

    await page.goto(`http://localhost:${harnessPort}/`, { waitUntil: 'networkidle' });
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

  it('main thread stays responsive (max rAF gap < 100ms) during 25-image detection', async () => {
    // Get image file list
    const imageFiles = readdirSync(FIXTURE_DIR)
      .filter((f) => f.endsWith('.png'))
      .sort();

    const imageUrls = imageFiles.map(
      (f) => `http://127.0.0.1:${fixturePort}/fixtures/synthetic-01/${f}`,
    );

    const boardConfig = {
      squaresX: 5,
      squaresY: 7,
      squareLength: 30,
      markerLength: 22,
      dictionaryId: 0,
      legacyPattern: false,
    };
    const flags = { fixK3: false, fixTangential: false, rationalModel: false, thinPrism: false };

    // Start rAF counter, run pipeline, collect frame timestamps
    const result = await page.evaluate(
      async ({ urls, board, fl }) => {
        const timestamps: number[] = [];
        let running = true;

        // rAF loop on main thread
        function tick() {
          timestamps.push(performance.now());
          if (running) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);

        // Run the real pipeline (detection + calibration in web worker)
        await window.plumbline.runWorker(urls, board, fl);

        running = false;
        // One more frame to capture final timestamp
        await new Promise((r) => requestAnimationFrame(r));

        // Compute max gap between adjacent frames
        let maxGap = 0;
        for (let i = 1; i < timestamps.length; i++) {
          const gap = timestamps[i] - timestamps[i - 1];
          if (gap > maxGap) maxGap = gap;
        }

        return {
          frameCount: timestamps.length,
          maxGapMs: maxGap,
          totalMs: timestamps[timestamps.length - 1] - timestamps[0],
        };
      },
      { urls: imageUrls, board: boardConfig, fl: flags },
    );

    console.log(`Worker responsiveness test:`);
    console.log(`  Frames: ${result.frameCount}`);
    console.log(`  Max rAF gap: ${result.maxGapMs.toFixed(1)} ms`);
    console.log(`  Total time: ${result.totalMs.toFixed(0)} ms`);

    expect(result.maxGapMs, `Max rAF gap ${result.maxGapMs.toFixed(1)}ms exceeds 100ms`).toBeLessThan(100);
    expect(result.frameCount).toBeGreaterThan(5); // Sanity: we should have many frames
  }, 300000);
});
