import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for kaggle-environments visualizer integration tests.
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testMatch: [
    'kaggle_environments/envs/*/visualizer/**/*.test.ts',
    'kaggle_environments/envs/open_spiel_env/games/*/visualizer/**/*.test.ts',
  ],
  fullyParallel: true,
  retries: 1,
  reporter: [['line'], ['json'], ['html']],
  projects: [
    {
      name: 'Visualizer Tests - Chrome',
      use: {
        trace: 'on-first-retry',
        ...devices['Desktop Chrome'],
      },
    },
  ],

  /* Run your local dev server before starting the tests */
  // webServer: {
  //   command: 'pnpm dev',
  //   url: 'http://127.0.0.1:3000',
  //   reuseExistingServer: !process.env.CI,
  // },
});
