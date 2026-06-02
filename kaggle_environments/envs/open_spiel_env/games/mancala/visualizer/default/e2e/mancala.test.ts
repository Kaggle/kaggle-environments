import { test, expect } from '@playwright/test';

test.describe('Mancala Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.mancala-root')).toBeVisible();
    await expect(page.locator('.mancala-board')).toBeVisible();
    await expect(page.locator('.mancala-title h1')).toContainText(/MANCALA/i);
    const playerCards = page.locator('.mancala-player-card');
    await expect(playerCards).toHaveCount(2);
    const stores = page.locator('.mancala-store');
    await expect(stores).toHaveCount(2);
  });

  test('displays correct game state at mid-game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    const midStep = Math.floor(parseInt(maxValue || '0') / 2);
    await slider.fill(String(midStep));
    await page.waitForTimeout(200);

    await expect(page.locator('.mancala-board')).toBeVisible();
    const pits = page.locator('.mancala-pit');
    await expect(pits).toHaveCount(12);
    // At least one player card should be highlighted as active mid-game
    await expect(page.locator('.mancala-player-card.is-active').first()).toBeVisible();
  });

  test('displays winner status at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    await expect(page.locator('.mancala-status').filter({ hasText: /Wins|draw/i })).toBeVisible();
  });
});
