import { defineConfig } from 'vite';
import checker from 'vite-plugin-checker';
import tsconfigPaths from 'vite-tsconfig-paths';
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';

export default defineConfig({
  base: './',
  optimizeDeps: {
    exclude: ['@kaggle-environments/core'],
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    cors: true,
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    cors: true,
  },
  plugins: [
    tsconfigPaths(),
    checker({ typescript: true }),
    // Inject CSS into JS bundle in production builds (matches dev behavior)
    cssInjectedByJsPlugin(),
    {
      name: 'custom-header-plugin',
      configureServer(server) {
        const originalPrintUrls = server.printUrls;
        server.printUrls = () => {
          const name = process.env.VITE_CUSTOM_HEADER_NAME;
          const path = process.env.VITE_CUSTOM_HEADER_PATH;
          if (name && path) {
            const header = `\n  ┃ Running: ${name}\n  ┃ Path:    ${path}\n`;
            process.stdout.write(header);
          }
          originalPrintUrls();
        };
      },
    },
  ],
});
