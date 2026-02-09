import { test, expect } from '@playwright/test';

test.describe('Halite Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game board canvases', async ({ page }) => {
    // Halite uses multiple canvases: buffer, background, foreground
    const foregroundCanvas = page.locator('canvas#foreground');
    await expect(foregroundCanvas).toBeVisible();

    const backgroundCanvas = page.locator('canvas#background');
    await expect(backgroundCanvas).toBeVisible();
  });
});
