import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/consistency.spec.ts', 'tests/diagnostics.spec.ts'],
    testTimeout: 300000,
    hookTimeout: 120000,
  },
});
