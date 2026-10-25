import { defineConfig } from 'vite';
import preact from '@preact/preset-vite';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    preact(),
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: 'auto',
      workbox: {
        // Cache all build assets + opencv.js (12 MB single-file WASM)
        globPatterns: ['**/*.{js,css,html,wasm,png,svg,webmanifest}'],
        maximumFileSizeToCacheInBytes: 15 * 1024 * 1024, // 15 MB for opencv.js
        runtimeCaching: [
          {
            urlPattern: /\/opencv\/.*\.js$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'opencv-cache',
              expiration: { maxAgeSeconds: 60 * 60 * 24 * 365 },
            },
          },
          {
            urlPattern: /^https:\/\/fonts\.(googleapis|gstatic)\.com\//,
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts',
              expiration: { maxEntries: 10, maxAgeSeconds: 60 * 60 * 24 * 365 },
            },
          },
        ],
      },
      manifest: {
        name: 'Plumbline',
        short_name: 'Plumbline',
        description: 'Free camera calibration, in your browser.',
        start_url: '/',
        display: 'standalone',
        background_color: '#ffffff',
        theme_color: '#E60012',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png' },
        ],
      },
    }),
  ],
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
    exclude: [
      'tests/consistency.spec.ts',
      'tests/worker-responsiveness.spec.ts',
      'tests/offline.spec.ts',
      'tests/large-image-timing.spec.ts',
      'node_modules/**',
    ],
  },
});
