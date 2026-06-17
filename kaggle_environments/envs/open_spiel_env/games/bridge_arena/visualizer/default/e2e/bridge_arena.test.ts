import { test, expect } from '@playwright/test';

test.describe('Bridge Arena Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();

    // Two team blocks in the header.
    const teamBlocks = page.locator('.header .team-block');
    expect(await teamBlocks.count()).toBe(2);
    await expect(page.locator('.header .team-block').first()).toContainText('Team A');
    await expect(page.locator('.header .team-block').last()).toContainText('Team B');

    // Contract panel and auction panel are present.
    await expect(page.locator('.panel-title').filter({ hasText: 'Contract' })).toBeVisible();
    await expect(page.locator('.panel-title').filter({ hasText: 'Auction' })).toBeVisible();

    await expect(page.locator('.status-container')).toBeVisible();
  });

  test('displays game state at mid-game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    const midStep = Math.floor(parseInt(maxValue || '0') / 2);
    await slider.fill(String(midStep));
    await page.waitForTimeout(200);

    // Mid-game: auction should have at least one call rendered.
    await expect(page.locator('.auction')).toBeVisible();
    await expect(page.locator('.status-container')).toBeVisible();
  });

  test('displays terminal status at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    await expect(page.locator('.status-container').filter({ hasText: /wins|Draw|Game over/i })).toBeVisible();
  });
});
