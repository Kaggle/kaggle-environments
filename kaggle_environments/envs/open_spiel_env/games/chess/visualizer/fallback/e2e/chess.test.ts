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

    const pieceImages = page.locator('div[id^="cell-"] img');
    await expect(pieceImages.first()).toBeVisible();
    const pieceCount = await pieceImages.count();
    expect(pieceCount).toBeGreaterThan(0);

    const chessTitle = page.locator('h1').filter({ hasText: 'Chess' });
    await expect(chessTitle).toBeVisible();
    const headerImages = page.locator('h1').locator('..').locator('img');
    const imgCount = await headerImages.count();
    expect(imgCount).toBeGreaterThanOrEqual(2);

    const statusText = page.locator('p').filter({ hasText: /Current Player|Winner|White|Black/ });
    await expect(statusText.first()).toBeVisible();
  });

  test('displays correct game state at mid-game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // Navigate to mid-game step
    const maxValue = await slider.getAttribute('max');
    const midStep = Math.floor(parseInt(maxValue || '0') / 2);
    await slider.fill(String(midStep));
    await page.waitForTimeout(200);

    const pieceImages = page.locator('div[id^="cell-"] img');
    await expect(pieceImages.first()).toBeVisible();
    const pieceCount = await pieceImages.count();
    expect(pieceCount).toBeGreaterThan(0);

    // Should show current player indicator (game not over yet)
    const statusText = page.locator('p').filter({ hasText: /Current Player|White|Black/ });
    await expect(statusText.first()).toBeVisible();
  });

  test('displays winner at end of game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    const winnerText = page.locator('p').filter({ hasText: /Wins|Draw/ });
    await expect(winnerText.first()).toBeVisible();
  });
});
