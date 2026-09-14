import { test, expect } from '@playwright/test';

test.describe('Hanabi Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();

    // Both teammates appear in the header. `toHaveCount` retries; a bare
    // `count()` can read the DOM before the first render lands.
    await expect(page.locator('.header .player')).toHaveCount(2);

    // The shared board: fireworks, both hands, the status line.
    await expect(page.locator('.fireworks')).toBeVisible();
    await expect(page.locator('.hand-panel')).toHaveCount(2);
    await expect(page.locator('.status-container')).toBeVisible();
  });

  test('displays correct game state at mid-game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    const midStep = Math.floor(parseInt(maxValue || '0') / 2);
    await slider.fill(String(midStep));
    await page.waitForTimeout(200);

    // Cards are dealt and the knowledge strip renders under each one.
    await expect(page.locator('.hb-card').first()).toBeVisible();
    await expect(page.locator('.hb-card .knowledge').first()).toBeVisible();

    // Team gauges (lives / info / deck / score) are present mid-game.
    await expect(page.locator('.gauge.score')).toBeVisible();
  });

  test('displays the team result at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    // Hanabi is cooperative -- there is no winner, only a shared score.
    await expect(page.locator('p').filter({ hasText: /Team scored|Out of lives|Perfect score/ })).toBeVisible();
  });

  test('renders a three-player game', async ({ page }) => {
    // Hanabi seats 2-5. The registered env pins two agents, so this replay was
    // generated from hanabi(players=3) and injected the same way as the
    // forfeit replay. It exercises hint targets of both +1 and +2, including
    // the wrap-around case where +2 from seat 2 resolves to seat 1.
    const injected = await page.evaluate(async (replayUrl) => {
      const resp = await fetch(replayUrl);
      if (!resp.ok) return { ok: false, status: resp.status };
      const replay = await resp.json();
      window.postMessage({ replay }, '*');
      return { ok: true, status: 200 };
    }, '/test-replay-3p.json');
    test.skip(!injected.ok, `test-replay-3p.json not available (HTTP ${injected.status})`);

    await page.waitForTimeout(300);

    // Three seats in the header and three hand panels on the board.
    await expect(page.locator('.header .player')).toHaveCount(3);
    await expect(page.locator('.hand-panel')).toHaveCount(3);

    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    await slider.fill(String(Math.floor(parseInt(maxValue || '0') / 2)));
    await page.waitForTimeout(200);

    // Every seat holds cards mid-game, so all three panels have faces.
    await expect(page.locator('.hand-panel').nth(2).locator('.hb-card').first()).toBeVisible();

    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);
    await expect(page.locator('p').filter({ hasText: /Team scored|Out of lives|Perfect score/ })).toBeVisible();
  });

  test('shows forfeit reason at final step', async ({ page }) => {
    // Swap in the forfeit replay via postMessage -- same dev server, no config
    // change. See replays/test-forfeit-replay.json.
    const injected = await page.evaluate(async (replayUrl) => {
      const resp = await fetch(replayUrl);
      if (!resp.ok) return { ok: false, status: resp.status };
      const replay = await resp.json();
      window.postMessage({ replay }, '*');
      return { ok: true, status: 200 };
    }, '/test-forfeit-replay.json');
    test.skip(!injected.ok, `test-forfeit-replay.json not available (HTTP ${injected.status})`);

    await page.waitForTimeout(300);
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    // Match loosely so any forfeit category (illegal move / timeout / error)
    // passes.
    await expect(
      page.getByText(/wins by default|forfeited|illegal move|ran out of time|failed to produce/i).first()
    ).toBeVisible();
  });
});
