import { test, expect } from '@playwright/test';

test.describe('Repeated Poker Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    const pokerTable = page.locator('.poker-table');
    await expect(pokerTable).toBeVisible();

    const communityCards = page.locator('.community-cards-container');
    await expect(communityCards).toBeVisible();
    const cardSlots = communityCards.locator('.card, .card-empty');
    const cardCount = await cardSlots.count();
    expect(cardCount).toBe(5);

    const potDisplay = page.locator('.pot-display');
    await expect(potDisplay).toBeVisible();
    await expect(potDisplay).toContainText(/Total Pot|Pot/i);

    const player0 = page.locator('.player-container-0');
    const player1 = page.locator('.player-container-1');
    await expect(player0).toBeVisible();
    await expect(player1).toBeVisible();

    const playerNames = page.locator('.player-name');
    await expect(playerNames.first()).toBeVisible();
    const nameCount = await playerNames.count();
    expect(nameCount).toBe(2);

    const playerCardAreas = page.locator('.player-card-area');
    await expect(playerCardAreas.first()).toBeVisible();
    const areaCount = await playerCardAreas.count();
    expect(areaCount).toBe(2);

    const stackValues = page.locator('.player-stack-value');
    await expect(stackValues.first()).toBeVisible();
    const stackCount = await stackValues.count();
    expect(stackCount).toBe(2);

    const dealerButton = page.locator('.dealer-button');
    await expect(dealerButton).toBeAttached();

    const legendTitle = page.locator('.legend-title');
    await expect(legendTitle).toBeVisible();
    await expect(legendTitle).toContainText(/Hand/i);

    const blindsInfo = page.locator('.legend-blinds');
    await expect(blindsInfo).toBeVisible();
    await expect(blindsInfo).toContainText(/Blinds/i);
  });

  test('displays correct game state at mid-game', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    // Navigate to mid-game step
    const maxValue = await slider.getAttribute('max');
    const midStep = Math.floor(parseInt(maxValue || '0') / 2);
    await slider.fill(String(midStep));
    await page.waitForTimeout(300);

    const pokerTable = page.locator('.poker-table');
    await expect(pokerTable).toBeVisible();

    // Player cards should be dealt and visible (not just empty/back)
    const visibleCards = page.locator('.card:not(.card-empty):not(.card-back)');
    const cardCount = await visibleCards.count();
    expect(cardCount).toBeGreaterThan(0);

    // Cards should have visible rank elements
    const rankElement = visibleCards.first().locator('.card-rank');
    await expect(rankElement).toBeVisible();

    // Pot should show an amount
    const potDisplay = page.locator('.pot-display');
    await expect(potDisplay).toBeVisible();

    // Legend table should show hand history
    const legendTable = page.locator('.legend-table');
    await expect(legendTable).toBeVisible();
    const headerRow = legendTable.locator('.legend-header');
    await expect(headerRow).toBeVisible();
  });

  test('displays game-over screen at end of match', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(300);

    const matchComplete = page.getByText('Match Complete');
    await expect(matchComplete).toBeVisible();
    const winnerText = page.getByText(/Winner:/);
    await expect(winnerText).toBeVisible();
  });
});
