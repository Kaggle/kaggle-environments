import { test, expect } from '@playwright/test';

test.describe('Repeated Poker Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the game', async ({ page }) => {
    // Poker table
    const pokerTable = page.locator('.poker-table');
    await expect(pokerTable).toBeVisible();

    // Community cards area with 5 card slots
    const communityCards = page.locator('.community-cards-container');
    await expect(communityCards).toBeVisible();
    const cardSlots = communityCards.locator('.card, .card-empty');
    const cardCount = await cardSlots.count();
    expect(cardCount).toBe(5);

    // Pot display
    const potDisplay = page.locator('.pot-display');
    await expect(potDisplay).toBeVisible();
    await expect(potDisplay).toContainText(/Total Pot|Pot/i);

    // Both player containers
    const player0 = page.locator('.player-container-0');
    const player1 = page.locator('.player-container-1');
    await expect(player0).toBeVisible();
    await expect(player1).toBeVisible();

    // Player names (2 players)
    const playerNames = page.locator('.player-name');
    await expect(playerNames.first()).toBeVisible();
    const nameCount = await playerNames.count();
    expect(nameCount).toBe(2);

    // Player card areas (2 players)
    const playerCardAreas = page.locator('.player-card-area');
    await expect(playerCardAreas.first()).toBeVisible();
    const areaCount = await playerCardAreas.count();
    expect(areaCount).toBe(2);

    // Player chip stacks (2 players)
    const stackValues = page.locator('.player-stack-value');
    await expect(stackValues.first()).toBeVisible();
    const stackCount = await stackValues.count();
    expect(stackCount).toBe(2);

    // Dealer button exists
    const dealerButton = page.locator('.dealer-button');
    await expect(dealerButton).toBeAttached();

    // Legend with hand history (use the one with .legend-title)
    const legendTitle = page.locator('.legend-title');
    await expect(legendTitle).toBeVisible();
    await expect(legendTitle).toContainText(/Hand/i);

    // Blinds information
    const blindsInfo = page.locator('.legend-blinds');
    await expect(blindsInfo).toBeVisible();
    await expect(blindsInfo).toContainText(/Blinds/i);
  });

  test('displays correct game state after navigation', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');

    // Navigate to step 2 where cards should be dealt
    if (maxValue && parseInt(maxValue) > 2) {
      await slider.fill('2');
      await page.waitForTimeout(200);

      // Cards should have rank and suit elements visible
      const cards = page.locator('.card:not(.card-empty):not(.card-back)');
      const cardCount = await cards.count();
      if (cardCount > 0) {
        const rankElement = cards.first().locator('.card-rank');
        await expect(rankElement).toBeVisible();
      }
    }

    // Navigate to final step
    await slider.fill(maxValue || '0');
    await page.waitForTimeout(300);

    // At game over, should show "Match Complete" and winner
    const matchComplete = page.getByText('Match Complete');
    await expect(matchComplete).toBeVisible();
    const winnerText = page.getByText(/Winner:/);
    await expect(winnerText).toBeVisible();
  });
});
