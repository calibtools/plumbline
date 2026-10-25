import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';

export default defineConfig({
  plugins: [preact()],
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    port: 5173,
  },
  test: {
    globals: true,
    environment: 'node',
    exclude: ['tests/consistency.spec.ts', 'node_modules/**'],
  },
});
