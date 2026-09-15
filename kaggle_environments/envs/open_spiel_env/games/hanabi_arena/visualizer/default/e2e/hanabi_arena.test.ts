import { test, expect } from '@playwright/test';

test.describe('Hanabi Arena Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();

    // Two teams in the header. `toHaveCount` retries; a bare `count()` can read
    // the DOM before the first render lands.
    await expect(page.locator('.header .team-pill')).toHaveCount(2);

    // Two tables side by side, four hands in total.
    await expect(page.locator('.table-column')).toHaveCount(2);
    await expect(page.locator('.fireworks')).toHaveCount(2);
    await expect(page.locator('.hand-panel')).toHaveCount(4);
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

    // Each table shows its own score independently.
    await expect(page.locator('.table-score')).toHaveCount(2);
  });

  test('reveals both teams hands, including the seats own cards', async ({ page }) => {
    // The merge is the load-bearing part of this visualizer: no single seat's
    // observation contains its own hand, so a face rendering in every panel is
    // what proves the four private views were combined. A failed merge would
    // leave the "?" placeholder on half the cards.
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    await slider.fill(String(Math.floor(parseInt((await slider.getAttribute('max')) || '0') / 2)));
    await page.waitForTimeout(200);

    for (let panel = 0; panel < 4; panel++) {
      await expect(page.locator('.hand-panel').nth(panel).locator('.hb-card .card-color').first()).toBeVisible();
    }
    await expect(page.locator('.hb-card .card-rank.unknown')).toHaveCount(0);
  });

  test('displays the head-to-head result at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(200);

    // Two teams play the same deal, so the ending is a win or a genuine draw.
    await expect(page.locator('p').filter({ hasText: /wins \d+-\d+|Draw —/ })).toBeVisible();
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
    // passes, but insist the offender's whole *team* is named -- a two-player
    // forfeit string here would be the wrong answer, not just a terser one.
    await expect(page.getByText(/Team \d+ \(.+&.+\) wins by default/i).first()).toBeVisible();

    // The Game Log picks one actor per step (the first with isTurn), so the
    // forfeiter has to be that actor or their reasoning never surfaces -- which
    // is exactly the step a reviewer wants to read. The list is virtualized, so
    // the last entry only enters the DOM once it is scrolled to.
    await page.getByText('Expand All').first().click();
    await page.locator('ul').last().hover();
    for (let i = 0; i < 6; i++) {
      await page.mouse.wheel(0, 600);
      await page.waitForTimeout(150);
    }
    await expect(page.getByText(/kept naming a slot that does not exist/i)).toBeVisible();
  });
});
