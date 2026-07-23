import { test, expect } from '@playwright/test';

// Note: `replays/test-forfeit-replay.json` exists on disk for manual dev use
// (`VITE_REPLAY_FILE=./replays/test-forfeit-replay.json pnpm dev`). A Shape 4
// forfeit test would need a second playwright project pointing the dev server
// at that replay file; none of the existing OpenSpiel visualizers wire this
// up yet, so we keep parity here.

test.describe('Capture the Flag Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();
    await expect(page.locator('.renderer-container canvas')).toBeVisible();

    const players = page.locator('.header .player');
    await expect(players.first()).toBeVisible();
    expect(await players.count()).toBe(2);

    await expect(page.locator('.status-container')).toBeVisible();
  });

  test('displays board state at mid-game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    const midStep = Math.floor(parseInt(maxValue || '0') / 2);
    await slider.fill(String(midStep));
    await page.waitForTimeout(200);

    await expect(page.locator('.renderer-container canvas')).toBeVisible();
    // Mid-game status shows a Turn indicator.
    await expect(page.locator('.status-container').filter({ hasText: /Turn/ })).toBeVisible();
  });

  test('displays winner status at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    await expect(page.locator('.status-container').filter({ hasText: /wins|Wins|Draw|Game over/ })).toBeVisible();
  });
});
