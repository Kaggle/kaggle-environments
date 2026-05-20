import { test, expect } from '@playwright/test';

test.describe('Backgammon Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();
    await expect(page.locator('.renderer-container canvas')).toBeVisible();

    // Both player cards are visible.
    const players = page.locator('.header .player');
    await expect(players.first()).toBeVisible();
    expect(await players.count()).toBe(2);

    // Status container is present.
    await expect(page.locator('.status-container')).toBeVisible();
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

    // Status is showing a turn indicator (mid-game is never terminal).
    await expect(page.locator('.status-container')).toContainText(/Turn:/);
  });

  test('displays winner status at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    await expect(page.locator('p').filter({ hasText: /Wins|Draw/ })).toBeVisible();
  });
});
