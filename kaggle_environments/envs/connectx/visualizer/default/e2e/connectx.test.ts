import { test, expect } from '@playwright/test';

test.describe('ConnectX Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game board canvas', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();
    await expect(page.locator('.renderer-container canvas')).toBeVisible();
  });

  test('displays winner status at final step', async ({ page }) => {
    // Navigate to the final step using the slider
    // The replay has 42 steps (0-41), so we set the slider to the max value
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '41');

    // Wait for the step to render
    await page.waitForTimeout(200);

    // At the final step, the status bar should display the winner
    await expect(page.locator('.status-bar')).toContainText('Winner');
  });
});
