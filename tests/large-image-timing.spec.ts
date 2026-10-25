/**
 * Large image (12 MP / 4000x3000) detection timing test.
 * NOT part of CI — run manually: npx vitest run --config vitest.timing.config.ts
 * Results go into docs/detection.md.
 */
import { describe, it, beforeAll, afterAll } from 'vitest';
import { chromium, type Browser, type Page } from 'playwright';
import { type ChildProcess, spawn } from 'child_process';
import { readFileSync, existsSync, readdirSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { createServer, type Server } from 'http';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');
const FIXTURE_DIR = join(PROJECT_ROOT, 'fixtures', 'synthetic-large');

describe('Large image detection timing', () => {
  let browser: Browser;
  let page: Page;
  let harnessProcess: ChildProcess;
  let fixtureServer: Server;
  let fixturePort: number;
  let harnessPort: number;

  beforeAll(async () => {
    if (!existsSync(FIXTURE_DIR)) {
      throw new Error('fixtures/synthetic-large/ not found');
    }

    fixtureServer = createServer((req, res) => {
      const url = new URL(req!.url!, 'http://localhost');
      const filePath = join(PROJECT_ROOT, url.pathname);
      try {
        const data = readFileSync(filePath);
        res.writeHead(200, { 'Content-Type': 'image/png', 'Access-Control-Allow-Origin': '*' });
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

    harnessPort = 15474 + Math.floor(Math.random() * 1000);
    harnessProcess = spawn('npx', ['vite', '--config', 'tests/harness/vite.config.ts', '--port', String(harnessPort), '--strictPort'], {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, VITE_CONFIG_NATIVE_IGNORE_WARNING: 'true' },
    });

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Vite did not start')), 30000);
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
      harnessProcess.on('exit', (code) => { clearTimeout(timeout); reject(new Error(`Vite exited ${code}`)); });
    });

    browser = await chromium.launch({ headless: true });
    page = await browser.newPage();
    await page.goto(`http://localhost:${harnessPort}/`, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.plumbline && window.plumbline.ready, { timeout: 60000 });
  }, 120000);

  afterAll(async () => {
    await browser?.close();
    harnessProcess?.kill();
    fixtureServer?.close();
  });

  it('measures per-image detection time on 4000x3000 images', async () => {
    const imageFiles = readdirSync(FIXTURE_DIR).filter((f) => f.endsWith('.png')).sort();

    const boardConfig = {
      squaresX: 5, squaresY: 7, squareLength: 30, markerLength: 22,
      dictionaryId: 0, legacyPattern: false,
    };
    const flags = { fixK3: false, fixTangential: false, rationalModel: false, thinPrism: false };

    console.log('\n=== 12 MP (4000x3000) single-image detection timing ===');
    const timings: number[] = [];

    for (const file of imageFiles) {
      const url = `http://127.0.0.1:${fixturePort}/fixtures/synthetic-large/${file}`;

      // Run single-image through the real pipeline, timing just the detection
      const ms = await page.evaluate(async ({ imgUrl, board, fl }) => {
        const t0 = performance.now();
        await window.plumbline.run([imgUrl], board, fl);
        const t1 = performance.now();
        return t1 - t0;
      }, { imgUrl: url, board: boardConfig, fl: flags });

      timings.push(ms);
      console.log(`  ${file}: ${ms.toFixed(0)} ms`);
    }

    const avg = timings.reduce((a, b) => a + b, 0) / timings.length;
    const max = Math.max(...timings);
    console.log(`  Average: ${avg.toFixed(0)} ms`);
    console.log(`  Max: ${max.toFixed(0)} ms`);

    if (max > 3000) {
      console.log('  ⚠️  Max exceeds 3 seconds — report to user for decision.');
    }
  }, 300000);
});
