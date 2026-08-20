import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  reporter: 'list',
  outputDir: '/tmp/test-results',
  use: {
    baseURL: 'http://localhost:5173',
    ...devices['Desktop Chrome'],
  },
  webServer: {
    command: 'pnpm dev',
    url: 'http://localhost:5173',
  },
});
