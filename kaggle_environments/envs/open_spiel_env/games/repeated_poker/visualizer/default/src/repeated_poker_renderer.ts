import poker_chip_1 from './images/poker_chip_1.svg';
import poker_chip_5 from './images/poker_chip_5.svg';
import poker_chip_10 from './images/poker_chip_10.svg';
import poker_chip_25 from './images/poker_chip_25.svg';
import poker_chip_100 from './images/poker_chip_100.svg';
import poker_card_back from './images/poker_card_back.svg';
import { RepeatedPokerStep, RepeatedPokerStepPlayer } from '@kaggle-environments/core';
import { acpcCardToDisplay, CardSuit, suitSVGs } from './components/utils';
import cssContent from './style.css?inline';

// Add property to global window object
declare global {
  interface Window {
    __poker_styles_injected?: boolean;
  }
}

/**
 * Options for the renderer
 */
interface RendererOptions {
  parent: HTMLElement;
  steps: RepeatedPokerStep[]; // This is the main data object
  step?: number;
  width: number;
  height: number;
}

/**
 * Interface for the cached DOM elements
 */
interface PokerTableElements {
  gameLayout: HTMLElement | null;
  pokerTableContainer: HTMLElement | null;
  pokerTable: HTMLElement | null;
  communityCardsContainer: HTMLElement | null;
  potDisplay: HTMLElement | null;
  playersContainer: HTMLElement | null;
  playerContainers: HTMLElement[];
  playerCardAreas: HTMLElement[];
  playerInfoAreas: HTMLElement[];
  playerNames: HTMLElement[];
  playerThumbnails: HTMLElement[];
  dealerButton: HTMLElement | null;
  chipStacks: HTMLElement[];
  diagnosticHeader: HTMLElement | null;
  stepCounter: HTMLElement | null; // Note: stepCounter is in 'elements' but not created in _ensurePokerTableElements
  legend: HTMLElement | null;
}

export function renderer(options: RendererOptions): void {
  const chipImages: Record<number, string> = {
    1: poker_chip_1,
    5: poker_chip_5,
    10: poker_chip_10,
    25: poker_chip_25,
    100: poker_chip_100,
  };

  const elements: PokerTableElements = {
    gameLayout: null,
    pokerTableContainer: null,
    pokerTable: null,
    communityCardsContainer: null,
    potDisplay: null,
    playersContainer: null,
    playerContainers: [],
    playerCardAreas: [],
    playerInfoAreas: [],
    playerNames: [],
    playerThumbnails: [],
    dealerButton: null,
    chipStacks: [],
    diagnosticHeader: null,
    stepCounter: null,
    legend: null,
  };

  function _injectStyles(passedOptions: Partial<RendererOptions>): void {
    if (typeof document === 'undefined') {
      return;
    }

    const styleId = 'data-poker-renderer-styles';
    const parentForStyles =
      passedOptions && passedOptions.parent ? passedOptions.parent.ownerDocument.head : document.head;

    if (!parentForStyles) {
      return;
    }

    // Find the existing style tag
    let style = parentForStyles.querySelector(`style[${styleId}]`);

    // If it doesn't exist, create it and append it
    if (!style) {
      style = document.createElement('style');
      style.setAttribute(styleId, 'true');
      parentForStyles.appendChild(style);
    }

    // 3. ALWAYS update the textContent
    style.textContent = cssContent;
  }

  function createCardElement(
    cardStr: string | null,
    isHidden: boolean = false,
    shouldHighlight: boolean = false
  ): HTMLElement {
    const cardDiv = document.createElement('div');
    cardDiv.classList.add('card');
    if (isHidden || !cardStr || cardStr === '?' || cardStr === '??') {
      cardDiv.classList.add('card-back');
      cardDiv.style.backgroundImage = `url(${poker_card_back})`;
      cardDiv.style.backgroundSize = 'cover';
      cardDiv.style.backgroundPosition = 'center';
    } else {
      const { rank, suit } = acpcCardToDisplay(cardStr);
      const rankSpan = document.createElement('span');
      rankSpan.classList.add('card-rank');
      rankSpan.textContent = rank;
      cardDiv.appendChild(rankSpan);

      const suitSpan = document.createElement('span');
      suitSpan.classList.add('card-suit');

      if (suitSVGs[suit as CardSuit]) {
        suitSpan.innerHTML = suitSVGs[suit as CardSuit];
      }

      cardDiv.appendChild(suitSpan);

      if (suit === 'hearts') cardDiv.classList.add('card-red');
      else if (suit === 'spades') cardDiv.classList.add('card-black');
      else if (suit === 'diamonds') cardDiv.classList.add('card-blue');
      else if (suit === 'clubs') cardDiv.classList.add('card-green');

      // Add highlight class if this card is part of the winning hand
      if (shouldHighlight) {
        cardDiv.classList.add('card-highlighted');
      }
    }
    return cardDiv;
  }

  function updateChipStack(chipStackElement: HTMLElement, betAmount: number): void {
    if (betAmount <= 0) {
      chipStackElement.style.display = 'none';
      return;
    }

    chipStackElement.style.display = 'flex';
    const chipsContainer = chipStackElement.querySelector('.chip-stack-chips') as HTMLElement;
    const labelElement = chipStackElement.querySelector('.chip-stack-label') as HTMLElement;

    if (!chipsContainer || !labelElement) return;

    chipsContainer.innerHTML = '';
    labelElement.textContent = String(betAmount);

    // Break down bet into denominations (100, 25, 10, 5, 1)
    const denominations = [100, 25, 10, 5, 1];
    let remaining = betAmount;
    const chipCounts: { denom: number; count: number }[] = [];

    for (const denom of denominations) {
      const count = Math.floor(remaining / denom);
      if (count > 0) {
        chipCounts.push({ denom, count: Math.min(count, 5) }); // Max 5 of each denomination
        remaining -= count * denom;
      }
    }

    // Render chips separated by denomination (highest to lowest, left to right)
    chipCounts.forEach(({ denom, count }) => {
      const denomStack = document.createElement('div');
      denomStack.className = 'chip-denomination-stack';

      for (let i = 0; i < count; i++) {
        const chip = document.createElement('div');
        chip.className = 'chip';
        const img = document.createElement('img');
        img.src = chipImages[denom];
        img.alt = `${denom} chip`;
        chip.appendChild(img);
        denomStack.appendChild(chip);
      }

      chipsContainer.appendChild(denomStack);
    });
  }

  // --- Board Parsing and Rendering ---
  function _ensurePokerTableElements(parentElement: HTMLElement): boolean {
    if (!parentElement) return false;
    parentElement.innerHTML = '';
    parentElement.classList.add('poker-renderer-host');

    elements.diagnosticHeader = document.createElement('h1');
    elements.diagnosticHeader.id = 'poker-renderer-diagnostic-header';
    elements.diagnosticHeader.textContent = 'Poker Table Initialized (Live Data)';
    elements.diagnosticHeader.style.cssText =
      'color: lime; background-color: black; padding: 5px; font-size: 12px; position: absolute; top: 0px; left: 0px; z-index: 10001; display: none;'; // Hidden by default
    parentElement.appendChild(elements.diagnosticHeader);

    elements.gameLayout = document.createElement('div');
    elements.gameLayout.className = 'poker-game-layout';
    parentElement.appendChild(elements.gameLayout);

    elements.pokerTableContainer = document.createElement('div');
    elements.pokerTableContainer.className = 'poker-table-container';
    elements.gameLayout.appendChild(elements.pokerTableContainer);

    elements.playersContainer = document.createElement('div');
    elements.playersContainer.className = 'players-container';
    elements.gameLayout.appendChild(elements.playersContainer);

    elements.pokerTable = document.createElement('div');
    elements.pokerTable.className = 'poker-table';
    elements.pokerTableContainer.appendChild(elements.pokerTable);

    const muckLine = document.createElement('div');
    muckLine.className = 'muck-line';
    elements.pokerTable.appendChild(muckLine);

    // Create chip stacks for each player inside the table
    elements.chipStacks = [];
    for (let i = 0; i < 2; i++) {
      const chipStack = document.createElement('div');
      chipStack.className = `chip-stack chip-stack-player${i}`;
      chipStack.style.display = 'none';
      chipStack.innerHTML = `
        <div class="chip-stack-chips"></div>
        <div class="chip-stack-label">0</div>
      `;
      elements.pokerTable.appendChild(chipStack);
      elements.chipStacks.push(chipStack);
    }

    const communityArea = document.createElement('div');
    communityArea.className = 'community-cards-area';
    elements.pokerTable.appendChild(communityArea);

    elements.potDisplay = document.createElement('div');
    elements.potDisplay.className = 'pot-display';
    communityArea.appendChild(elements.potDisplay);

    elements.communityCardsContainer = document.createElement('div');
    elements.communityCardsContainer.className = 'community-cards-container';
    communityArea.appendChild(elements.communityCardsContainer);

    elements.playerContainers = [];
    elements.playerCardAreas = [];
    elements.playerInfoAreas = [];
    elements.playerNames = [];
    elements.playerThumbnails = [];

    for (let i = 0; i < 2; i++) {
      // Create player container that groups all player elements
      const playerContainer = document.createElement('div');
      playerContainer.className = `player-container player-container-${i}`;
      elements.playersContainer.appendChild(playerContainer);
      elements.playerContainers.push(playerContainer);

      // Player name wrapper with thumbnail
      const playerNameWrapper = document.createElement('div');
      playerNameWrapper.className = `player-name-wrapper`;
      playerContainer.appendChild(playerNameWrapper);

      // Player thumbnail
      const playerThumbnail = document.createElement('img');
      playerThumbnail.className = `player-thumbnail`;
      playerThumbnail.style.display = 'none'; // Hidden by default
      playerNameWrapper.appendChild(playerThumbnail);
      elements.playerThumbnails.push(playerThumbnail);

      // Player name
      const playerName = document.createElement('div');
      playerName.className = `player-name`;
      playerName.textContent = `Player ${i}`;
      playerNameWrapper.appendChild(playerName);
      elements.playerNames.push(playerName);

      // Info area containing bet, cards, and stack
      const playerInfoArea = document.createElement('div');
      playerInfoArea.className = `player-info-area`;
      playerInfoArea.innerHTML = `
            <div class="bet-display"></div>
            <div class="stack-cards-wrapper">
              <div class="player-card-area">
                <div class="player-cards-container"></div>
              </div>
              <div class="player-stack">
                <span class="player-stack-value">0</span>
              </div>
            </div>
            <div class="player-stats-container">
              <div class="player-hand-rank"></div>
              <div class="player-odds"></div>
            </div>
            `;
      playerContainer.appendChild(playerInfoArea);
      elements.playerInfoAreas.push(playerInfoArea);

      // Get reference to card area (already in DOM)
      const playerCardArea = playerInfoArea.querySelector('.player-card-area') as HTMLElement;
      elements.playerCardAreas.push(playerCardArea);
    }

    elements.dealerButton = document.createElement('div');
    elements.dealerButton.className = 'dealer-button';
    elements.dealerButton.textContent = 'D';
    elements.dealerButton.style.display = 'none';
    elements.playersContainer.appendChild(elements.dealerButton);

    elements.legend = document.createElement('div');
    elements.legend.className = 'legend';
    elements.legend.innerHTML = `
      <div class="legend-title"></div>
      <div class="legend-body"></div>
    `;
    elements.gameLayout.appendChild(elements.legend);

    return true;
  } // --- State Parsing ---

  function _applyScale(parentElement: HTMLElement): void {
    if (!parentElement || !elements.gameLayout) return;

    const parentWidth = parentElement.clientWidth;
    const parentHeight = parentElement.clientHeight;

    const baseWidth = 1000;
    const baseHeight = 1000;

    const scaleX = parentWidth / baseWidth;
    const scaleY = parentHeight / baseHeight;
    const scale = Math.min(scaleX, scaleY);

    elements.gameLayout.style.transform = `scale(${scale})`;
  }

  // Helper to find the shortest unique name by skipping common prefixes
  function _getDistinguishingNameMap(names: string[]): Map<string, string> {
    const map = new Map<string, string>();
    if (names.length === 0) return map;
    // If only one name, just take the first word as before
    if (names.length === 1) {
      map.set(names[0], names[0].split(' ')[0]);
      return map;
    }

    const splitNames = names.map((n) => n.split(' '));
    let diffIndex = 0;
    const maxWords = Math.max(...splitNames.map((s) => s.length));

    // Step through words until we find one that doesn't match across all players
    for (let i = 0; i < maxWords; i++) {
      const firstVal = splitNames[0][i];
      // Check if ALL players have the exact same word at this index
      const allMatch = splitNames.every((parts) => parts[i] === firstVal && parts[i] !== undefined);
      if (!allMatch) {
        diffIndex = i;
        break;
      }
    }

    // Build the map using the word at the differentiating index
    names.forEach((fullName, i) => {
      // Fallback to the full name if the split name is too short to have a word at diffIndex
      const shortName = splitNames[i][diffIndex] || fullName;
      map.set(fullName, shortName);
    });

    return map;
  }

  interface CompletedHand {
    handNum: number;
    winnerName: string;
    winnerThumbnail?: string;
    amount: number;
  }

  interface DerivedLeaderboardInfo {
    topLeader: { name: string; winnings: number; thumbnail?: string } | null;
    completedHands: CompletedHand[];
  }

  function _deriveLeaderboardData(steps: RepeatedPokerStep[], currentStepIndex: number): DerivedLeaderboardInfo {
    const playerCumulativeRewards = new Map<number, number>();
    const playerDetails = new Map<number, { name: string; thumbnail?: string }>();
    const completedHands: CompletedHand[] = [];

    // Identify unique hands up to the current step
    const relevantSteps = steps.slice(0, currentStepIndex + 1);
    const hands = new Map<number, RepeatedPokerStep>();

    // We only need the *last* step of any given hand to see its final result/rewards
    relevantSteps.forEach((step) => {
      hands.set(step.currentHandIndex, step);
    });

    // Iterate through finished hands to build history and totals
    hands.forEach((lastStepOfHand, handIndex) => {
      const lastStepOfHandPlayers = lastStepOfHand.players as RepeatedPokerStepPlayer[];
      // Assume a hand is "complete" if any player has a non-null reward
      const isHandComplete = lastStepOfHandPlayers.some((p) => p.reward !== null);

      if (isHandComplete) {
        // Update cumulative totals for ALL players in this hand
        lastStepOfHandPlayers.forEach((p) => {
          // Save player details for later lookups
          if (!playerDetails.has(p.id)) {
            playerDetails.set(p.id, { name: p.name, thumbnail: p.thumbnail });
          }

          const currentTotal = playerCumulativeRewards.get(p.id) || 0;
          // Ensure we don't add null/undefined rewards
          playerCumulativeRewards.set(p.id, currentTotal + (p.reward || 0));
        });

        // Add to hand history table
        // Find the winner(s). There might be multiple in a split pot.
        const winners = lastStepOfHandPlayers.filter((p) => p.isWinner);
        winners.forEach((winner) => {
          completedHands.push({
            handNum: handIndex + 1,
            winnerName: winner.name,
            winnerThumbnail: winner.thumbnail,
            amount: winner.reward || 0,
          });
        });
      }
    });

    // Find the current leader
    let topLeader = null;
    let maxWinnings = -Infinity;

    playerCumulativeRewards.forEach((winnings, playerId) => {
      if (winnings > maxWinnings) {
        maxWinnings = winnings;
        const details = playerDetails.get(playerId);
        topLeader = {
          name: details?.name || 'Unknown',
          thumbnail: details?.thumbnail,
          winnings: winnings,
        };
      }
    });

    return {
      topLeader,
      completedHands: completedHands.sort((a, b) => a.handNum - b.handNum),
    };
  }

  function _renderLegendUI(steps: RepeatedPokerStep[], currentStepIndex: number): void {
    if (!elements.legend || !steps || !steps[currentStepIndex]) return;

    const legendTitle = elements.legend.querySelector('.legend-title') as HTMLElement;
    const legendBody = elements.legend.querySelector('.legend-body') as HTMLElement;
    if (!legendTitle || !legendBody) return;

    const currentStepData = steps[currentStepIndex];
    const { currentHandIndex } = currentStepData;
    // Calculate derived data specifically for this frame
    const { topLeader, completedHands } = _deriveLeaderboardData(steps, currentStepIndex);

    // Gather ALL unique player full names encountered so far for disambiguation
    const allPlayerNames = new Set<string>();
    if (topLeader) allPlayerNames.add(topLeader.name);
    completedHands.forEach((h) => allPlayerNames.add(h.winnerName));
    // Also add current players to ensure complete set if they haven't won yet
    currentStepData.players.forEach((p) => allPlayerNames.add(p.name));

    // 2. Generate the short name map
    const shortNameMap = _getDistinguishingNameMap(Array.from(allPlayerNames));

    // --- RENDER TITLE SECTION ---
    legendTitle.innerHTML = '';

    const handSpan = document.createElement('span');
    handSpan.textContent = `Hand: ${currentHandIndex != null ? currentHandIndex + 1 : 'Standby'}`;
    legendTitle.appendChild(handSpan);

    if (topLeader) {
      const leaderInfoDiv = document.createElement('div');
      leaderInfoDiv.className = 'legend-leader-info';

      if (topLeader.thumbnail) {
        const leaderThumbnail = document.createElement('img');
        leaderThumbnail.src = topLeader.thumbnail;
        leaderThumbnail.className = 'legend-title-avatar';
        leaderInfoDiv.appendChild(leaderThumbnail);
      }

      const leaderNameSpan = document.createElement('span');
      const leaderShortName = shortNameMap.get(topLeader.name) || topLeader.name;
      leaderNameSpan.textContent = `${leaderShortName} is up ${topLeader.winnings}`;
      leaderInfoDiv.appendChild(leaderNameSpan);
      legendTitle.appendChild(leaderInfoDiv);
    }

    // --- RENDER BODY/TABLE SECTION ---
    legendBody.innerHTML = '';

    const table = document.createElement('div');
    table.className = 'legend-table';

    const headerRow = document.createElement('div');
    headerRow.className = 'legend-row legend-header';
    ['Hand', 'Winner', 'Amount'].forEach((text) => {
      const cell = document.createElement('div');
      cell.className = 'legend-cell';
      cell.textContent = text;
      headerRow.appendChild(cell);
    });
    table.appendChild(headerRow);

    if (completedHands.length > 0) {
      // Slice and reverse to show newest hands first
      completedHands
        .slice()
        .reverse()
        .forEach((hand) => {
          const row = document.createElement('div');
          row.className = 'legend-row';

          const handCell = document.createElement('div');
          handCell.className = 'legend-cell';
          handCell.textContent = hand.handNum.toString();
          row.appendChild(handCell);

          const winnerCell = document.createElement('div');
          winnerCell.className = 'legend-cell';
          const winnerCellContainer = document.createElement('div');
          winnerCellContainer.className = 'legend-winner-cell';

          if (hand.winnerThumbnail) {
            const winnerThumbnail = document.createElement('img');
            winnerThumbnail.src = hand.winnerThumbnail;
            winnerThumbnail.className = 'legend-avatar';
            winnerCellContainer.appendChild(winnerThumbnail);
          }

          const winnerNameSpan = document.createElement('span');
          winnerNameSpan.textContent = shortNameMap.get(hand.winnerName) || hand.winnerName;
          winnerCellContainer.appendChild(winnerNameSpan);

          winnerCell.appendChild(winnerCellContainer);
          row.appendChild(winnerCell);

          const amountCell = document.createElement('div');
          amountCell.className = 'legend-cell';
          amountCell.textContent = hand.amount.toString();
          row.appendChild(amountCell);

          table.appendChild(row);
        });
    } else {
      const emptyRow = document.createElement('div');
      emptyRow.className = 'legend-row';
      const emptyCell = document.createElement('div');
      emptyCell.className = 'legend-cell';
      emptyCell.style.textAlign = 'center';
      emptyCell.textContent = '-';
      // We used 3 columns in the header
      emptyCell.style.gridColumn = '1 / span 3';
      emptyRow.appendChild(emptyCell);
      table.appendChild(emptyRow);
    }
    legendBody.appendChild(table);
  }

  function _renderPokerTableUI(currentStepData: RepeatedPokerStep): void {
    if (!elements.pokerTable || !currentStepData) return;

    const { players, communityCards, stepType, pot, winOdds, bestFiveCardHands, bestHandRankTypes } = currentStepData;

    if (!elements.communityCardsContainer || !elements.potDisplay) return;

    elements.communityCardsContainer.innerHTML = '';
    // Always show 5 slots for the river
    // Display cards left to right, with empty slots at the end
    const numCommunityCards = 5;

    // TODO: [TYPE_MISMATCH] 'communityCards' is a string, but the code expects an array of card strings - move this to the transformer
    const communityCardsArray = communityCards.match(/.{1,2}/g) || ([] as string[]);
    const numCards = communityCardsArray.length;

    // Get winning player's best hand for highlighting (only on final step with all 5 community cards)
    const isShowdown = numCards === 5 && stepType === 'final';
    const winnerIndex = players.findIndex((p) => (p as RepeatedPokerStepPlayer).isWinner);
    const winnerBestHand =
      winnerIndex !== -1 && isShowdown && bestFiveCardHands?.[winnerIndex]
        ? bestFiveCardHands[winnerIndex].match(/.{1,2}/g) || ([] as string[])
        : ([] as string[]);

    // Add actual cards
    for (let i = 0; i < numCards; i++) {
      const shouldHighlight = winnerBestHand.includes(communityCardsArray[i]);
      elements.communityCardsContainer.appendChild(createCardElement(communityCardsArray[i], false, shouldHighlight));
    }

    // Fill remaining slots with empty cards
    for (let i = numCards; i < numCommunityCards; i++) {
      const emptyCard = document.createElement('div');
      emptyCard.classList.add('card', 'card-empty');
      elements.communityCardsContainer.appendChild(emptyCard);
    }

    elements.potDisplay.textContent = `Total Pot : ${pot}`;

    players.forEach((basePlayerData, index) => {
      // The JS code expects properties from 'RepeatedPokerStepPlayer'.
      // Casting 'basePlayerData' to 'RepeatedPokerStepPlayer' to access properties.
      const playerData = basePlayerData as RepeatedPokerStepPlayer;

      const playerNameElement = elements.playerNames[index];
      if (playerNameElement) {
        playerNameElement.textContent = playerData.name;

        if (playerData.isTurn) {
          playerNameElement.classList.add('current-turn');
        } else {
          playerNameElement.classList.remove('current-turn');
        }

        // Add winner class if player won
        if (playerData.isWinner) {
          playerNameElement.classList.add('winner');
        } else {
          playerNameElement.classList.remove('winner');
        }
      }

      // Update thumbnail
      const playerThumbnailElement = elements.playerThumbnails[index];
      if (playerThumbnailElement && playerData.thumbnail) {
        (playerThumbnailElement as HTMLImageElement).src = playerData.thumbnail;
        playerThumbnailElement.style.display = 'block';
      } else if (playerThumbnailElement) {
        playerThumbnailElement.style.display = 'none';
      }

      // Update card area (left side)
      const playerCardArea = elements.playerCardAreas[index];
      if (playerCardArea) {
        const playerCardsContainer = playerCardArea.querySelector('.player-cards-container') as HTMLElement;
        if (!playerCardsContainer) return;
        playerCardsContainer.innerHTML = '';

        // In heads-up, we show both hands at the end.
        const showCards = playerData.cards && !playerData.cards.includes(null!);

        // TODO: [TYPE_MISMATCH] 'playerData.cards' is a string, but code expects an array - move this to the transformer
        const playerCardsArray = playerData.cards ? playerData.cards.match(/.{1,2}/g) : [null, null];

        // Parse the best hand for this player to determine which cards to highlight (only on showdown)
        const bestHandArray =
          bestFiveCardHands && bestFiveCardHands[index]
            ? bestFiveCardHands[index].match(/.{1,2}/g) || ([] as string[])
            : ([] as string[]);
        const shouldHighlightWinningHand = playerData.isWinner && showCards && isShowdown && bestHandArray.length > 0;

        (playerCardsArray || [null, null]).forEach((cardStr) => {
          const shouldHighlight = shouldHighlightWinningHand && cardStr && bestHandArray.includes(cardStr);
          playerCardsContainer.appendChild(
            createCardElement(cardStr, !showCards && cardStr !== null, !!shouldHighlight)
          );
        });
      }

      // Update chip stacks on the table
      if (elements.chipStacks[index]) {
        updateChipStack(elements.chipStacks[index], playerData.currentBet);
      }

      // Update info area (right side)
      const playerInfoArea = elements.playerInfoAreas[index];
      if (playerInfoArea) {
        // Highlight active player's pod
        if (playerData.isTurn) {
          playerInfoArea.classList.add('active-player');
        } else {
          playerInfoArea.classList.remove('active-player');
        }

        // Highlight winner's pod
        if (playerData.isWinner) {
          playerInfoArea.classList.add('winner-player');
        } else {
          playerInfoArea.classList.remove('winner-player');
        }

        const stackValueEl = playerInfoArea.querySelector('.player-stack-value') as HTMLElement;
        if (stackValueEl) {
          stackValueEl.textContent = `${playerData.chipStack}`;
        }

        const betDisplay = playerInfoArea.querySelector('.bet-display') as HTMLElement;
        if (betDisplay) {
          if (playerData.isWinner) {
            betDisplay.classList.add('winner-player');
          }
          if (playerData.currentBet > 0) {
            if (playerData.actionDisplayText) {
              betDisplay.textContent = playerData.actionDisplayText;
            } else {
              betDisplay.textContent = '';
            }
            betDisplay.style.display = 'block';
          }
        }

        const handRankElement = playerInfoArea.querySelector('.player-hand-rank') as HTMLElement;
        if (handRankElement && bestHandRankTypes && bestHandRankTypes[index]) {
          handRankElement.textContent = bestHandRankTypes[index];
        } else if (handRankElement) {
          handRankElement.textContent = '';
        }

        const playerOddsElement = playerInfoArea.querySelector('.player-odds') as HTMLElement;
        // WinOdds is an array like [P0WinOdds, P0TieOdds, P1WinOdds, P1TieOdds] - our index is either 0 or 1, so
        // for the P1 case just set the index to 2
        const winOddsIndex = index === 0 ? 0 : 2;
        if (playerOddsElement && winOdds && winOdds[winOddsIndex] != undefined) {
          const winOddsStringForPlayer = `WIN: ${winOdds[winOddsIndex].toLocaleString(undefined, {
            style: 'percent',
            minimumFractionDigits: 2,
          })}`;
          const tieOddsStringForPlayer = `TIE: ${winOdds[winOddsIndex + 1].toLocaleString(undefined, {
            style: 'percent',
            minimumFractionDigits: 2,
          })}`;

          const oddsString = `${winOddsStringForPlayer} · ${tieOddsStringForPlayer}`;

          playerOddsElement.textContent = oddsString;
        } else if (playerOddsElement) {
          playerOddsElement.textContent = '';
        }
      }
    });

    const dealerPlayerIndex = players.findIndex((p) => (p as RepeatedPokerStepPlayer).isDealer);
    if (elements.dealerButton && elements.playersContainer) {
      if (dealerPlayerIndex !== -1) {
        elements.dealerButton.style.display = 'block';
        elements.dealerButton.classList.remove('dealer-player0', 'dealer-player1');
        elements.dealerButton.classList.add(`dealer-player${dealerPlayerIndex}`);

        const playerInfoArea = elements.playerInfoAreas[dealerPlayerIndex];
        if (playerInfoArea) {
          const boxRect = playerInfoArea.getBoundingClientRect();
          const containerRect = elements.playersContainer.getBoundingClientRect();
          const left = boxRect.left - containerRect.left - elements.dealerButton.offsetWidth - 20;
          elements.dealerButton.style.left = `${left}px`;
        }
      } else {
        elements.dealerButton.style.display = 'none';
      }
    }
  }

  // --- MAIN EXECUTION LOGIC ---
  const { parent } = options;
  if (!parent) {
    console.error('Renderer: Parent element not provided.');
    return;
  }

  _injectStyles(options);

  if (!_ensurePokerTableElements(parent)) {
    console.error('Renderer: Failed to ensure poker table elements.');
    parent.innerHTML = '<p style="color:red;">Error: Could not create poker table structure.</p>';
    return;
  }

  _renderPokerTableUI(options.steps[options.step ?? 0]);
  _renderLegendUI(options.steps, options.step ?? 0);

  // Apply initial scale
  _applyScale(parent);

  // Watch for container size changes and reapply scale
  if (typeof ResizeObserver !== 'undefined') {
    const resizeObserver = new ResizeObserver(() => {
      _applyScale(parent);
    });
    resizeObserver.observe(parent);
  }
}
