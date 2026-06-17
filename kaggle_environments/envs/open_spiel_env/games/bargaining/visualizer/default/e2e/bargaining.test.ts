import { test, expect } from '@playwright/test';

test.describe('Bargaining Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();
    await expect(page.locator('.brg-header')).toBeVisible();
    await expect(page.locator('.brg-pool')).toBeVisible();
    await expect(page.locator('.brg-log')).toBeVisible();
    await expect(page.locator('.brg-status')).toBeVisible();
    expect(await page.locator('.brg-player-card').count()).toBe(2);
  });

  test('displays correct game state at mid-game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    const midStep = Math.floor(parseInt(maxValue || '0') / 2);
    await slider.fill(String(midStep));
    await page.waitForTimeout(200);

    await expect(page.locator('.brg-pool')).toBeVisible();
    // At least one offer bubble should have appeared by mid-game.
    await expect(page.locator('.brg-bubble').first()).toBeVisible();
  });

  test('displays terminal status at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    await expect(page.locator('.brg-status').filter({ hasText: /Deal accepted|No agreement|wins|tied/ })).toBeVisible();
  });
});
