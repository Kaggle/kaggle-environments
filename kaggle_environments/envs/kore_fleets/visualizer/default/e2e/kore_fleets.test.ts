import { test, expect } from '@playwright/test';

test.describe('Kore Fleets Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game board canvases', async ({ page }) => {
    // Kore Fleets uses multiple canvases similar to Halite
    const foregroundCanvas = page.locator('canvas#foreground');
    await expect(foregroundCanvas).toBeVisible();

    const backgroundCanvas = page.locator('canvas#background');
    await expect(backgroundCanvas).toBeVisible();
  });

  test('displays playback controls', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await expect(slider).toBeVisible();
  });

  test('displays step counter', async ({ page }) => {
    // The viewer shows step count like "1 / 400"
    const stepCounter = page.getByText(/\d+\s*\/\s*\d+/);
    await expect(stepCounter).toBeVisible();
  });

  test('step slider navigates through game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 10) {
      await slider.fill('10');
      await page.waitForTimeout(200);
    }

    // Foreground canvas should still be visible
    const canvas = page.locator('canvas#foreground');
    await expect(canvas).toBeVisible();
  });

  test('game state updates at different steps', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // Navigate to mid-game
    const maxValue = await slider.getAttribute('max');
    if (maxValue) {
      const midStep = Math.floor(parseInt(maxValue) / 2);
      await slider.fill(String(midStep));
      await page.waitForTimeout(200);
    }

    // Foreground canvas should still be rendering
    const canvas = page.locator('canvas#foreground');
    await expect(canvas).toBeVisible();
  });
});
