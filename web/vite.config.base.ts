import { defineConfig } from 'vite';
import checker from 'vite-plugin-checker';
import tsconfigPaths from 'vite-tsconfig-paths';
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';

// Support custom port via environment variable (used by Playwright tests)
const port = process.env.VITE_PORT ? parseInt(process.env.VITE_PORT, 10) : 5173;

export default defineConfig({
  base: './',
  optimizeDeps: {
    exclude: ['@kaggle-environments/core'],
  },
  server: {
    host: '0.0.0.0',
    port,
    cors: true,
  },
  preview: {
    host: '0.0.0.0',
    port,
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
