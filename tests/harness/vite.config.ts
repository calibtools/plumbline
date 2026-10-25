import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: resolve(import.meta.dirname!),
  publicDir: resolve(import.meta.dirname!, '../../public'),
  build: {
    outDir: resolve(import.meta.dirname!, 'dist'),
    emptyOutDir: true,
  },
});
