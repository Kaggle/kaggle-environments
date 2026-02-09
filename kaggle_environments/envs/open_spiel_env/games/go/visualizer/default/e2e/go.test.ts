import { test, expect } from '@playwright/test';

test.describe('Go Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game board', async ({ page }) => {
    // The Go renderer creates an SVG element for the board
    // Wait for the board title to confirm the visualizer has rendered
    const title = page.locator('h1');
    await expect(title).toBeVisible();
    await expect(title).toContainText(/Go/);
  });

  test('displays board title with dimensions', async ({ page }) => {
    // The Go renderer shows a title like "Go (9x9)" or "Go (19x19)"
    const title = page.locator('h1');
    await expect(title).toBeVisible();
    await expect(title).toContainText(/Go \(\d+×\d+\)/);
  });

  test('displays coordinate labels', async ({ page }) => {
    // Go board has column labels (A, B, C, etc. - skipping I)
    // These appear as text in the page
    await expect(page.getByText('A').first()).toBeVisible();
    await expect(page.getByText('B').first()).toBeVisible();
    // Row numbers
    await expect(page.getByText('1').first()).toBeVisible();
  });

  test('displays status information with komi', async ({ page }) => {
    // The status shows Komi value for Go games
    const komiText = page.getByText(/Komi/);
    await expect(komiText).toBeVisible();
  });

  test('displays last move information', async ({ page }) => {
    // Navigate to a step where a move has been made
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 1) {
      await slider.fill('1');
      await page.waitForTimeout(200);
    }

    // Should show last move information
    const lastMoveText = page.getByText(/Last move/);
    await expect(lastMoveText).toBeVisible();
  });

  test('displays current player', async ({ page }) => {
    // The status shows whose turn it is (Black or White)
    const playerText = page.locator('p').filter({ hasText: /Black|White/ });
    await expect(playerText.first()).toBeVisible();
  });

  test('step slider works', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // Move to a later step
    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 5) {
      await slider.fill('5');
      await page.waitForTimeout(200);
    }

    // Page should still be functional
    const title = page.locator('h1');
    await expect(title).toBeVisible();
  });
});
