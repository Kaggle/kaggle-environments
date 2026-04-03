import { test, expect, Page } from '@playwright/test';

/**
 * Shared EpisodePlayer + ReasoningLogs tests.
 *
 * These exercise the game-agnostic shell: UI modes, dense mode,
 * playback controls, and ReasoningLogs behavior. They run against
 * the core dev server with a PlaceholderRenderer so they are
 * independent of any specific game visualizer.
 */

/** Wait for the EpisodePlayer to finish loading and show content. */
async function waitForPlayer(page: Page) {
  await page.getByRole('heading', { name: 'Game Log' }).waitFor({ timeout: 10_000 });
}

async function pauseIfPlaying(page: Page) {
  const pauseBtn = page.getByRole('button', { name: 'Pause' });
  if (await pauseBtn.isVisible().catch(() => false)) {
    await pauseBtn.click();
  }
  // Wait for UI to settle
  await page.waitForTimeout(100);
}

const muiSlider = 'input[type="range"][name="Change Step"]';
const muiSpeedSelect = '[aria-label="Change Playback Speed"]';

test.describe('EpisodePlayer — side-panel mode', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForPlayer(page);
  });

  test('renders ReasoningLogs sidebar with controls', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Game Log' })).toBeVisible();

    await expect(page.getByRole('button', { name: 'Restart' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Previous Step' })).toBeVisible();
    await expect(page.getByRole('button', { name: /Play|Pause/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Next Step' })).toBeVisible();
    await expect(page.getByTestId('step-counter')).toBeVisible();

    await expect(page.locator(muiSlider)).toBeVisible();
    await expect(page.locator(muiSpeedSelect)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Streaming View' })).toBeVisible();
    await expect(page.getByRole('button', { name: /Expand All|Collapse All/ })).toBeVisible();
  });

  test('does not render inline PlaybackControls', async ({ page }) => {
    // Inline mode uses a plain <input type="range"> with aria-label "Step slider"
    await expect(page.locator('[aria-label="Step slider"]')).toBeHidden();
  });

  test('renders step cards in the log', async ({ page }) => {
    const stepCards = page.locator('[role="button"]');
    await expect(stepCards.first()).toBeVisible();
  });
});

test.describe('ReasoningLogs — playback controls', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForPlayer(page);
    await pauseIfPlaying(page);
  });

  test('play/pause toggles button icon', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Play|Pause/ });
    await expect(btn).toHaveAttribute('aria-label', 'Play');

    await btn.click();
    await expect(btn).toHaveAttribute('aria-label', 'Pause');

    await btn.click();
    await expect(btn).toHaveAttribute('aria-label', 'Play');
  });

  test('next/previous step buttons change the step counter', async ({ page }) => {
    // Restart and pause to get to a known state (step 1)
    await page.getByRole('button', { name: 'Restart' }).click();
    await page.waitForTimeout(100);
    await pauseIfPlaying(page);

    const counter = page.getByTestId('step-counter');

    await expect(counter).toBeVisible();
    const initialText = await counter.textContent();
    const initialStep = parseInt(initialText!.split('/')[0]);

    await page.getByRole('button', { name: 'Next Step' }).click();
    await expect(counter).toHaveText(new RegExp(`^${initialStep + 1}/`));

    await page.getByRole('button', { name: 'Previous Step' }).click();
    await expect(counter).toHaveText(new RegExp(`^${initialStep}/`));
  });

  test('restart button goes back to step 1', async ({ page }) => {
    // Advance a few steps
    await page.getByRole('button', { name: 'Next Step' }).click();
    await page.getByRole('button', { name: 'Next Step' }).click();

    // Restart (starts playback), then pause
    await page.getByRole('button', { name: 'Restart' }).click();
    await page.waitForTimeout(100);
    await pauseIfPlaying(page);

    const counter = page.getByText(/^\d+\/\d+$/);
    await expect(counter).toHaveText(/^1\//);
  });

  test('speed selector opens and has options', async ({ page }) => {
    await page.locator(muiSpeedSelect).click();

    await expect(page.getByRole('option', { name: /Speed 2/ })).toBeVisible();
    await expect(page.getByRole('option', { name: /Speed 0\.5/ })).toBeVisible();

    await page.getByRole('option', { name: /Speed 2/ }).click();
    await expect(page.getByRole('option', { name: /Speed 2/ })).toBeHidden();
  });
});

test.describe('ReasoningLogs — mode and expansion', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForPlayer(page);
  });

  test('toggles between condensed and streaming view', async ({ page }) => {
    const modeButton = page.getByRole('button', { name: /Streaming View|Log View/ });
    await expect(modeButton).toHaveText(/Streaming View/);

    await modeButton.click();
    await expect(modeButton).toHaveText(/Log View/);

    await modeButton.click();
    await expect(modeButton).toHaveText(/Streaming View/);
  });

  test('expand/collapse all button toggles and is only visible in condensed mode', async ({ page }) => {
    const expandBtn = page.getByRole('button', { name: /Expand All|Collapse All/ });
    await expect(expandBtn).toBeVisible();
    await expect(expandBtn).toHaveText(/Expand All/);

    await expandBtn.click();
    await expect(expandBtn).toHaveText(/Collapse All/);

    await expandBtn.click();
    await expect(expandBtn).toHaveText(/Expand All/);

    // Switch to streaming mode — expand/collapse disappears
    await page.getByRole('button', { name: 'Streaming View' }).click();
    await expect(expandBtn).toBeHidden();

    // Switch back — it reappears
    await page.getByRole('button', { name: 'Log View' }).click();
    await expect(page.getByRole('button', { name: /Expand All|Collapse All/ })).toBeVisible();
  });

  test('show/hide thinking buttons work on step cards', async ({ page }) => {
    await pauseIfPlaying(page);

    const thinkingButtons = page.locator('button:has-text("Show thinking"), button:has-text("Hide thinking")');
    await thinkingButtons.first().waitFor({ timeout: 5_000 });

    const firstBtn = thinkingButtons.first();

    // In condensed mode, thinking is collapsed by default
    await expect(firstBtn).toHaveText(/Show thinking/);

    await firstBtn.click();
    await expect(firstBtn).toHaveText(/Hide thinking/);

    // Toggle back
    await firstBtn.click();
    await expect(firstBtn).toHaveText(/Show thinking/);
  });
});

test.describe('ReasoningLogs — close and reopen panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForPlayer(page);
  });

  test('close button hides panel and shows Game Log button', async ({ page }) => {
    await page.getByRole('button', { name: 'Collapse Episodes' }).click();

    await expect(page.getByRole('heading', { name: 'Game Log' })).toBeHidden();

    const reopenBtn = page.getByRole('button', { name: 'Game Log' });
    await expect(reopenBtn).toBeVisible();

    await reopenBtn.click();
    await expect(page.getByRole('heading', { name: 'Game Log' })).toBeVisible();
  });
});

test.describe('EpisodePlayer — dense mode (desktop)', () => {
  test('visualizer renders in default (non-dense) mode', async ({ page }) => {
    await page.goto('/');
    await waitForPlayer(page);

    // The PlaceholderRenderer shows game name
    const visualizer = page.locator('h2:has-text("Visualizer")');
    await expect(visualizer).toBeVisible();
  });
});

test.describe('EpisodePlayer — mobile viewport', () => {
  test.use({ viewport: { width: 600, height: 800 } });

  test('core playback buttons are present on small viewport', async ({ page }) => {
    await page.goto('/');
    await waitForPlayer(page);

    // On mobile-width (below tablet breakpoint of 840px), layout stacks vertically.
    // Core playback buttons should still be visible.
    await expect(page.getByRole('button', { name: 'Restart' })).toBeVisible();
    await expect(page.getByRole('button', { name: /Play|Pause/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Next Step' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Previous Step' })).toBeVisible();
    await expect(page.getByTestId('step-counter')).toBeVisible();
  });

  test('full controls (slider, speed, mode toggle) are visible on mobile when not dense', async ({ page }) => {
    await page.goto('/');
    await waitForPlayer(page);

    await expect(page.locator(muiSlider)).toBeVisible();
    await expect(page.locator(muiSpeedSelect)).toBeVisible();
    await expect(page.getByRole('button', { name: /Streaming View|Log View/ })).toBeVisible();
  });
});

test.describe('EpisodePlayer — keyboard shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForPlayer(page);
    await pauseIfPlaying(page);
  });

  test('Space toggles play/pause', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Play|Pause/ });
    await expect(btn).toHaveAttribute('aria-label', 'Play');

    // Click the visualizer area first to ensure focus is not on a button
    // (keyboard handler ignores events on buttons/inputs)
    await page.locator('h2:has-text("test-game Visualizer")').click();
    await page.keyboard.press('Space');
    await expect(btn).toHaveAttribute('aria-label', 'Pause');

    await page.keyboard.press('Space');
    await expect(btn).toHaveAttribute('aria-label', 'Play');
  });

  test('Arrow keys change steps', async ({ page }) => {
    const counter = page.getByText(/^\d+\/\d+$/);
    const initialText = await counter.textContent();
    const initialStep = parseInt(initialText!.split('/')[0]);

    // Click the visualizer area to ensure focus is not on a button
    await page.locator('h2:has-text("test-game Visualizer")').click();

    await page.keyboard.press('ArrowRight');
    await expect(counter).toHaveText(new RegExp(`^${initialStep + 1}/`));

    await page.keyboard.press('ArrowLeft');
    await expect(counter).toHaveText(new RegExp(`^${initialStep}/`));
  });
});
