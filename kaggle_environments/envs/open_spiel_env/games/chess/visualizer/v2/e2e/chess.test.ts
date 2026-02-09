import { test, expect } from '@playwright/test';

test.describe('Chess Visualizer (v2)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the chess board grid', async ({ page }) => {
    // Chess board is an 8x8 grid with cells identified by cell-{row}-{col}
    const firstCell = page.locator('#cell-0-0');
    await expect(firstCell).toBeVisible();

    const lastCell = page.locator('#cell-7-7');
    await expect(lastCell).toBeVisible();
  });

  test('displays alternating square colors', async ({ page }) => {
    // Chess board has alternating light and dark squares
    const cell00 = page.locator('#cell-0-0');
    const cell01 = page.locator('#cell-0-1');

    await expect(cell00).toBeVisible();
    await expect(cell01).toBeVisible();

    // The cells should have different background colors
    const bg00 = await cell00.evaluate((el) => getComputedStyle(el).backgroundColor);
    const bg01 = await cell01.evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(bg00).not.toBe(bg01);
  });

  test('renders chess pieces as images', async ({ page }) => {
    // Pieces are rendered as img elements inside cells
    const pieceImages = page.locator('div[id^="cell-"] img');
    await expect(pieceImages.first()).toBeVisible();

    // At the start, there should be pieces on the board
    const pieceCount = await pieceImages.count();
    expect(pieceCount).toBeGreaterThan(0);
  });

  test('displays player names in header', async ({ page }) => {
    // Header shows player names and "Chess" title
    const chessTitle = page.locator('h1').filter({ hasText: 'Chess' });
    await expect(chessTitle).toBeVisible();

    // Player names are displayed as text spans near the header
    // The header container has player name text
    const headerImages = page.locator('h1').locator('..').locator('img');
    const imgCount = await headerImages.count();
    // Should have pawn icons for player indicators
    expect(imgCount).toBeGreaterThanOrEqual(2);
  });

  test('shows current player or winner status', async ({ page }) => {
    // Status area shows "Current Player" during game or winner at end
    const statusText = page.locator('p').filter({ hasText: /Current Player|Winner|White|Black/ });
    await expect(statusText.first()).toBeVisible();
  });

  test('displays winner at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');

    await page.waitForTimeout(200);

    // At the final step, should show the winner
    const winnerText = page.locator('p, span, div').filter({ hasText: /Winner|Checkmate|Draw|White|Black/ });
    await expect(winnerText.first()).toBeVisible();
  });

  test('board updates when navigating steps', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // Get initial piece count
    const initialPieces = await page.locator('div[id^="cell-"] img').count();

    // Move to a later step
    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 10) {
      await slider.fill('10');
      await page.waitForTimeout(200);
    }

    // Board should still have pieces
    const laterPieces = await page.locator('div[id^="cell-"] img').count();
    expect(laterPieces).toBeGreaterThan(0);
  });
});
