import { test, expect } from '@playwright/test';

test.describe('Chess Visualizer (Default)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    // 8x8 board grid
    const firstCell = page.locator('#cell-0-0');
    const lastCell = page.locator('#cell-7-7');
    await expect(firstCell).toBeVisible();
    await expect(lastCell).toBeVisible();

    // Alternating square colors
    const cell00 = page.locator('#cell-0-0');
    const cell01 = page.locator('#cell-0-1');
    const bg00 = await cell00.evaluate((el) => getComputedStyle(el).backgroundColor);
    const bg01 = await cell01.evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(bg00).not.toBe(bg01);

    // Chess pieces as images
    const pieceImages = page.locator('div[id^="cell-"] img');
    await expect(pieceImages.first()).toBeVisible();
    const pieceCount = await pieceImages.count();
    expect(pieceCount).toBeGreaterThan(0);

    // Title and player indicators
    const chessTitle = page.locator('h1').filter({ hasText: 'Chess' });
    await expect(chessTitle).toBeVisible();
    const headerImages = page.locator('h1').locator('..').locator('img');
    const imgCount = await headerImages.count();
    expect(imgCount).toBeGreaterThanOrEqual(2);

    // Current player status
    const statusText = page.locator('p').filter({ hasText: /Current Player|Winner|White|Black/ });
    await expect(statusText.first()).toBeVisible();
  });

  test('displays correct game state after navigation', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // Navigate to final step
    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    // At the final step, should show the winner
    const winnerText = page.locator('p, span, div').filter({ hasText: /Winner|Checkmate|Draw|White|Black/ });
    await expect(winnerText.first()).toBeVisible();
  });
});
