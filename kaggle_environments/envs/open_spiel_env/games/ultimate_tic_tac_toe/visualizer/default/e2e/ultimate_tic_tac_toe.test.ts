import { test, expect } from '@playwright/test';

test.describe('Ultimate Tic-Tac-Toe Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[BROWSER CONSOLE] ${msg.type()}: ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`[BROWSER ERROR] ${err.message}\n${err.stack}`));
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();
    await expect(page.locator('.renderer-container canvas')).toBeVisible();

    // Check player indicators in header
    const players = page.locator('.header .player');
    await expect(players.first()).toBeVisible();
    expect(await players.count()).toBe(2);

    // Status container is present
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

    // Status is showing a turn indicator
    await expect(page.locator('.status-container')).toContainText(/turn/i);
  });

  test('displays winner status at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    // End state will show won/draw text
    await expect(page.locator('.status-container')).toContainText(/wins|draw/i);
  });
});
