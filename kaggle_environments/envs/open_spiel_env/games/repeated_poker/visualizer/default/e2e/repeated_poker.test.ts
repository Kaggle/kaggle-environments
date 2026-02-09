import { test, expect } from '@playwright/test';

test.describe('Repeated Poker Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders the poker table', async ({ page }) => {
    const pokerTable = page.locator('.poker-table');
    await expect(pokerTable).toBeVisible();
  });

  test('displays community cards area', async ({ page }) => {
    const communityCards = page.locator('.community-cards-container');
    await expect(communityCards).toBeVisible();

    // Community cards area should have card slots (5 for Texas Hold'em)
    const cards = communityCards.locator('.card, .card-empty');
    const cardCount = await cards.count();
    expect(cardCount).toBe(5);
  });

  test('displays pot amount', async ({ page }) => {
    const potDisplay = page.locator('.pot-display');
    await expect(potDisplay).toBeVisible();
    await expect(potDisplay).toContainText(/Total Pot|Pot/i);
  });

  test('renders player containers for both players', async ({ page }) => {
    const player0 = page.locator('.player-container-0');
    const player1 = page.locator('.player-container-1');

    await expect(player0).toBeVisible();
    await expect(player1).toBeVisible();
  });

  test('displays player names', async ({ page }) => {
    const playerNames = page.locator('.player-name');
    await expect(playerNames.first()).toBeVisible();

    const nameCount = await playerNames.count();
    expect(nameCount).toBe(2);
  });

  test('shows player cards', async ({ page }) => {
    const playerCardAreas = page.locator('.player-card-area');
    await expect(playerCardAreas.first()).toBeVisible();

    // Each player should have a card area
    const areaCount = await playerCardAreas.count();
    expect(areaCount).toBe(2);
  });

  test('displays player chip stacks', async ({ page }) => {
    const stackValues = page.locator('.player-stack-value');
    await expect(stackValues.first()).toBeVisible();

    // Should show chip counts for both players
    const stackCount = await stackValues.count();
    expect(stackCount).toBe(2);
  });

  test('renders dealer button', async ({ page }) => {
    const dealerButton = page.locator('.dealer-button');
    // Dealer button may or may not be visible depending on game state
    // Just verify the element exists in DOM
    await expect(dealerButton).toBeAttached();
  });

  test('displays hand history legend', async ({ page }) => {
    const legend = page.locator('.legend');
    await expect(legend).toBeVisible();

    // Legend should have a title showing hand number
    const legendTitle = legend.locator('.legend-title');
    await expect(legendTitle).toContainText(/Hand/i);
  });

  test('shows blinds information', async ({ page }) => {
    const blindsInfo = page.locator('.legend-blinds');
    await expect(blindsInfo).toBeVisible();
    await expect(blindsInfo).toContainText(/Blinds/i);
  });

  test('renders cards with suits and ranks', async ({ page }) => {
    // Navigate to a step where cards are dealt
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 2) {
      await slider.fill('2');
      await page.waitForTimeout(200);
    }

    // Cards should have rank and suit elements
    const cards = page.locator('.card:not(.card-empty):not(.card-back)');
    const cardCount = await cards.count();

    if (cardCount > 0) {
      const rankElement = cards.first().locator('.card-rank');
      await expect(rankElement).toBeVisible();
    }
  });

  test('displays chip stacks on table during betting', async ({ page }) => {
    // Navigate to a step with betting
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 3) {
      await slider.fill('3');
      await page.waitForTimeout(200);
    }

    // Chip stacks may be visible when players have bet
    const chipStacks = page.locator('.chip-stack');
    // Just verify the elements exist (may be hidden if no bets)
    await expect(chipStacks.first()).toBeAttached();
  });

  test('shows game-over screen at final step', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });
    const maxValue = await slider.getAttribute('max');
    await slider.fill(maxValue || '0');

    await page.waitForTimeout(300);

    // At game over, should show final screen with "Match Complete" and winner info
    const matchComplete = page.getByText('Match Complete');
    await expect(matchComplete).toBeVisible();

    // Should also show winner
    const winnerText = page.getByText(/Winner:/);
    await expect(winnerText).toBeVisible();
  });

  test('legend table shows hand history', async ({ page }) => {
    // Navigate to later in the game where hands have been played
    const slider = page.locator('input[type="range"]');
    await slider.waitFor({ state: 'visible' });

    const maxValue = await slider.getAttribute('max');
    if (maxValue && parseInt(maxValue) > 20) {
      await slider.fill('20');
      await page.waitForTimeout(200);
    }

    const legendTable = page.locator('.legend-table');
    await expect(legendTable).toBeVisible();

    // Should have header row
    const headerRow = legendTable.locator('.legend-header');
    await expect(headerRow).toBeVisible();
  });
});
