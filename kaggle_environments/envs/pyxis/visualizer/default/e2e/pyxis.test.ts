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

  test('displays the outcome at the final step', async ({ page }) => {
    await scrubTo(page, 1);
    await expect(page.locator('.status-container')).toContainText(/wins|Draw/i);
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
