import { test, expect } from '@playwright/test';

async function scrubTo(page: import('@playwright/test').Page, fraction: number) {
  const slider = page.locator('input[type="range"]');
  await slider.waitFor({ state: 'visible' });
  const max = parseInt((await slider.getAttribute('max')) || '0', 10);
  await slider.fill(String(Math.floor(max * fraction)));
  await page.waitForTimeout(200);
}

test.describe('Pyxis Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    await expect(page.locator('.renderer-container')).toBeVisible();
    await expect(page.locator('canvas.pipeline')).toBeVisible();
    await expect(page.locator('.player-card')).toHaveCount(2);
  });

  test('displays portfolio state at mid-game', async ({ page }) => {
    await scrubTo(page, 0.5);
    await expect(page.locator('.player-card').first()).toContainText(/£/);
    await expect(page.locator('.panel-title').filter({ hasText: /Intelligence/i })).toBeVisible();
  });

  test('shows the trial bill and runway that decide solvency', async ({ page }) => {
    await scrubTo(page, 0.5);
    await expect(page.locator('.player-commitment').first()).toContainText(/trials £\S+\/step\s+·\s+\d+ steps of cash/);
  });

  test('counts only running trials against sites', async ({ page }) => {
    await scrubTo(page, 0.5);
    const text = (await page.locator('.player-sites').first().textContent()) ?? '';
    const [, inTrials, free, owned] = text.match(/(\d+) in trials .* (\d+)\/(\d+) sites free/) ?? [];
    expect(Number(free)).toBe(Number(owned) - Number(inTrials));
  });

  test("reports each player's moves for the step", async ({ page }) => {
    // The rules bot buys PTRS readings from the first step.
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    await slider.fill('2');
    await page.waitForTimeout(200);
    await expect(page.locator('.player-moves').first()).toContainText(/reading/);
  });

  test('prices each BD offer', async ({ page }) => {
    await scrubTo(page, 0.2);
    await expect(page.locator('.bd-panel')).toContainText(/PTRS \d\.\d\d · eNPV/);
  });

  test('flags the clinical site auction on the steps it is open', async ({ page }) => {
    // The auction opens every 20 steps; step 10 of a 100-step match is one.
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    await slider.fill('10');
    await page.waitForTimeout(200);
    await expect(page.locator('.auction-banner.open')).toBeVisible();
    await slider.fill('11');
    await page.waitForTimeout(200);
    await expect(page.locator('.auction-banner.open')).toHaveCount(0);
  });

  test('displays the outcome at the final step', async ({ page }) => {
    await scrubTo(page, 1);
    await expect(page.locator('.status-container')).toContainText(/rules_bot wins/);
  });
});

test.describe('Pyxis Visualizer - forfeit', () => {
  test('reports the illegal action instead of freezing mid-game', async ({ page }) => {
    await page.goto('/');
    // Swap in the forfeit replay via postMessage: same dev server, no config change.
    const injected = await page.evaluate(async (replayUrl) => {
      const resp = await fetch(replayUrl);
      if (!resp.ok) return { ok: false, status: resp.status };
      window.postMessage({ replay: await resp.json() }, '*');
      return { ok: true, status: 200 };
    }, '/test-forfeit-replay.json');
    test.skip(!injected.ok, `test-forfeit-replay.json not available (HTTP ${injected.status})`);

    await page.waitForTimeout(300);
    await scrubTo(page, 1);
    await expect(page.getByText(/illegal action|wins by default/i).first()).toBeVisible();
  });
});
