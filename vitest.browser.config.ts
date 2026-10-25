import { defineConfig } from 'vitest/config';

/** Config for all browser-based tests (consistency, worker, offline) */
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: [
      'tests/consistency.spec.ts',
      'tests/worker-responsiveness.spec.ts',
      'tests/offline.spec.ts',
    ],
    testTimeout: 300000,
    hookTimeout: 120000,
  },
});
