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
});
