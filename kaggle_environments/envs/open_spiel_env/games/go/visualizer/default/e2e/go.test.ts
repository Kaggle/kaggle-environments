import { test, expect } from '@playwright/test';

test.describe('Go Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('Load game', async ({ page }) => {
    await page.getByLabel('Pause').click();

    const slider = page.locator('input[type="range"]');
    const maxValue = await slider.getAttribute('max');
    expect(Number(maxValue)).toBeGreaterThan(5);

    const versusBanner = page.getByTestId('ribbon');
    await versusBanner.waitFor({ state: 'attached' });
    await expect(versusBanner).toContainText(' vs. ');
  });

  test('Mid game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    const maxValue = await slider.getAttribute('max');
    const midStep = Math.round(Number(maxValue) / 4) * 2; // always white
    await slider.fill(String(midStep));
    await page.waitForTimeout(300);

    const currentPlayer = page.locator('[class*="_active_"]');
    const currentIcon = currentPlayer.getByAltText('White');
    await expect(currentIcon).toBeVisible();
  });

  test('Game over', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    const maxValue = await slider.getAttribute('max');
    const maxStep = Number(maxValue);
    await slider.fill(String(maxStep));

    const gameOver = page.getByLabel('Game Over');
    await gameOver.waitFor({ state: 'attached' });
    await expect(gameOver).toContainText(/Winner is|It's a draw/);
  });
});
