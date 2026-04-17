import { test, expect } from '@playwright/test';

test.describe('Werewolf Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('page loads without errors', async ({ page }) => {
    // Just verify the app container exists and page loaded
    // Werewolf uses WebGL and complex rendering, so we keep tests minimal
    const app = page.locator('#app');
    await expect(app).toBeVisible();
  });
});
