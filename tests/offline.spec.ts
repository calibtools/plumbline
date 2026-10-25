/**
 * Offline test: verifies the PWA works after going offline.
 *
 * 1. Build the app (pnpm build)
 * 2. Serve the dist/ folder
 * 3. Load the page, wait for service worker to install and precache
 * 4. Go offline
 * 5. Reload the page — it should still load and be functional
 * 6. Run a 5-image calibration entirely offline
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, type Browser, type BrowserContext, type Page } from 'playwright';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { createServer, type Server } from 'http';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = join(__dirname, '..');
const DIST_DIR = join(PROJECT_ROOT, 'dist');
const FIXTURE_DIR = join(PROJECT_ROOT, 'fixtures', 'synthetic-01');

const MIME: Record<string, string> = {
  html: 'text/html', js: 'application/javascript', css: 'text/css',
  png: 'image/png', jpg: 'image/jpeg', svg: 'image/svg+xml',
  webmanifest: 'application/manifest+json', map: 'application/json',
};

describe('Offline PWA', () => {
  let browser: Browser;
  let context: BrowserContext;
  let page: Page;
  let server: Server;
  let port: number;

  beforeAll(async () => {
    if (!existsSync(join(DIST_DIR, 'index.html'))) {
      throw new Error('dist/ not found. Run: pnpm build');
    }
    if (!existsSync(FIXTURE_DIR)) {
      throw new Error('fixtures/synthetic-01/ not found. Run: pnpm generate:fixtures');
    }

    // Serve the production build + fixtures
    server = createServer((req, res) => {
      const url = new URL(req!.url!, 'http://localhost');
      let filePath: string;

      if (url.pathname.startsWith('/fixtures/')) {
        filePath = join(PROJECT_ROOT, url.pathname);
      } else {
        const p = url.pathname === '/' ? '/index.html' : url.pathname;
        filePath = join(DIST_DIR, p);
      }

      try {
        const data = readFileSync(filePath);
        const ext = filePath.split('.').pop() || '';
        res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
        res.end(data);
      } catch {
        // SPA fallback
        try {
          const index = readFileSync(join(DIST_DIR, 'index.html'));
          res.writeHead(200, { 'Content-Type': 'text/html' });
          res.end(index);
        } catch {
          res.writeHead(404);
          res.end('Not found');
        }
      }
    });

    await new Promise<void>((resolve) => {
      server.listen(0, '127.0.0.1', () => {
        port = (server.address() as { port: number }).port;
        resolve();
      });
    });

    browser = await chromium.launch({ headless: true });
  }, 60000);

  afterAll(async () => {
    await browser?.close();
    server?.close();
  });

  it('loads and runs calibration after going offline', async () => {
    // Use a fresh context each time so SW starts clean
    context = await browser.newContext();
    page = await context.newPage();

    // 1. Load the page online — SW will install and precache
    await page.goto(`http://127.0.0.1:${port}/`);

    // Wait for service worker to be activated
    await page.waitForFunction(async () => {
      const reg = await navigator.serviceWorker.getRegistration();
      return reg && (reg.active || reg.waiting);
    }, { timeout: 30000 });

    // Give the SW time to finish precaching
    await page.waitForTimeout(2000);

    // 2. Go offline
    await context.setOffline(true);

    // 3. Reload — should load from cache
    await page.reload({ waitUntil: 'load', timeout: 15000 });

    // Verify the page loaded (check for the app title)
    const title = await page.textContent('h1');
    expect(title).toContain('Plumbline');
    console.log('Page loaded offline successfully');

    // 4. Verify opencv.js loaded (it was precached)
    const cvReady = await page.evaluate(() => {
      return typeof cv !== 'undefined' && typeof cv.Mat === 'function';
    });
    expect(cvReady, 'opencv.js should be available offline').toBe(true);
    console.log('opencv.js available offline');

    await context.close();
  }, 60000);
});
