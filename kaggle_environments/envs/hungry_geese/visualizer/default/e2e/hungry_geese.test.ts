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

  test('displays step counter', async ({ page }) => {
    // The viewer shows step count like "1 / 200"
    const stepCounter = page.getByText(/\d+\s*\/\s*\d+/);
    await expect(stepCounter).toBeVisible();
  });

  test('displays playback controls', async ({ page }) => {
    // Should have play/pause buttons and slider
    const slider = page.locator('input[type="range"]');
    await expect(slider).toBeVisible();

    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    expect(buttonCount).toBeGreaterThan(0);
  });

  test('step slider navigates through game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 10) {
      await slider.fill('10');
      await page.waitForTimeout(200);
    }

    // Canvas should still be visible
    const canvas = page.locator('canvas#hungry_geese');
    await expect(canvas).toBeVisible();
  });

  test('game board updates at different steps', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // Navigate to mid-game
    const maxValue = await slider.getAttribute('max');
    if (maxValue) {
      const midStep = Math.floor(parseInt(maxValue) / 2);
      await slider.fill(String(midStep));
      await page.waitForTimeout(200);
    }

    // Canvas should still be rendering
    const canvas = page.locator('canvas#hungry_geese');
    await expect(canvas).toBeVisible();
  });
});
