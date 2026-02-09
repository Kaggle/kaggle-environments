import { test, expect } from '@playwright/test';

test.describe('LLM 20 Questions Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game canvas', async ({ page }) => {
    // LLM 20 Questions renders all game content on canvas
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();
  });

  test('displays playback controls', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await expect(slider).toBeVisible();
  });

  test('step slider navigates through rounds', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 5) {
      await slider.fill('5');
      await page.waitForTimeout(200);
    }

    // Canvas should still be visible after navigation
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();
  });
});
