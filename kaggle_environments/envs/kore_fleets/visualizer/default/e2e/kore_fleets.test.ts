import { test, expect } from '@playwright/test';

test.describe('Kore Fleets Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game board canvases', async ({ page }) => {
    const foregroundCanvas = page.locator('canvas#foreground');
    await expect(foregroundCanvas).toBeVisible();

    const backgroundCanvas = page.locator('canvas#background');
    await expect(backgroundCanvas).toBeVisible();
  });
});
