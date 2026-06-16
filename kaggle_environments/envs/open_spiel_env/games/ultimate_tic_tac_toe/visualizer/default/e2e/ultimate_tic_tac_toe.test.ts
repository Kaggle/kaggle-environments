import { test, expect } from '@playwright/test';

test.describe('Ultimate Tic-Tac-Toe Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();
    await expect(page.locator('.renderer-container canvas')).toBeVisible();

    // Check legend items
    const legendItems = page.locator('.player-legend .legend-item');
    await expect(legendItems.first()).toBeVisible();
    expect(await legendItems.count()).toBe(2);

    // Status bar is present
    await expect(page.locator('.status-bar')).toBeVisible();
  });

  test('displays correct game state at mid-game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    const midStep = Math.floor(parseInt(maxValue || '0') / 2);
    await slider.fill(String(midStep));
    await page.waitForTimeout(200);

    // Board still rendered.
    await expect(page.locator('.renderer-container canvas')).toBeVisible();

    // Status is showing a turn indicator
    await expect(page.locator('.status-bar')).toContainText(/turn/i);
  });

  test('displays winner status at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    // End state will show won/draw text
    await expect(page.locator('.status-bar')).toContainText(/won the match|ended in a draw/i);
  });
});
