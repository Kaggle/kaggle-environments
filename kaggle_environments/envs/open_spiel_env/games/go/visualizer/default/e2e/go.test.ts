import { test, expect } from '@playwright/test';

test.describe('Go Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    // Title with board dimensions (e.g., "Go (9×9)")
    const title = page.locator('h1');
    await expect(title).toBeVisible();
    await expect(title).toContainText(/Go \(\d+×\d+\)/);

    // Coordinate labels (columns A, B, etc. and row numbers)
    await expect(page.getByText('A').first()).toBeVisible();
    await expect(page.getByText('B').first()).toBeVisible();
    await expect(page.getByText('1').first()).toBeVisible();

    // Status information with komi
    const komiText = page.getByText(/Komi/);
    await expect(komiText).toBeVisible();

    // Current player indicator (Black or White)
    const playerText = page.locator('p').filter({ hasText: /Black|White/ });
    await expect(playerText.first()).toBeVisible();
  });

  test('displays correct game state after navigation', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // Navigate to step 1 where a move has been made
    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 1) {
      await slider.fill('1');
      await page.waitForTimeout(200);
    }

    // After a move, should show last move information
    const lastMoveText = page.getByText(/Last move/);
    await expect(lastMoveText).toBeVisible();
  });
});
