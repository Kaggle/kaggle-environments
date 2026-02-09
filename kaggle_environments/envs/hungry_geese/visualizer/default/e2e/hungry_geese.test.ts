import { test, expect } from '@playwright/test';

test.describe('Hungry Geese Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game board canvas', async ({ page }) => {
    // Hungry Geese uses canvas for rendering
    const canvas = page.locator('canvas#hungry_geese');
    await expect(canvas).toBeVisible();
  });

  test('renders buffer canvas for sprites', async ({ page }) => {
    // The renderer creates a buffer canvas for drawing sprites
    const bufferCanvas = page.locator('canvas#buffer');
    await expect(bufferCanvas).toBeAttached();
  });
});
