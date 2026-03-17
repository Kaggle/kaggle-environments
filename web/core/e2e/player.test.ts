import { test, expect } from '@playwright/test';

test.describe('EpisodePlayer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('autoplays on load', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // The slider should advance past step 0 without any user interaction
    await expect(slider).not.toHaveValue('0', { timeout: 5000 });

    // The play/pause button should show "Pause" (meaning playback is active)
    await expect(page.getByRole('button', { name: 'Pause' })).toBeVisible();
  });
});
