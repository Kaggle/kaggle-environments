import poker_chip_1 from './images/poker_chip_1.svg';
import poker_chip_5 from './images/poker_chip_5.svg';
import poker_chip_10 from './images/poker_chip_10.svg';
import poker_chip_25 from './images/poker_chip_25.svg';
import poker_chip_100 from './images/poker_chip_100.svg';
import { RepeatedPokerStep, RepeatedPokerStepPlayer } from '@kaggle-environments/core';
import { acpcCardToDisplay, CardSuit, suitSVGs } from './components/utils';
import cssContent from "./style.css?inline";

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
    if (typeof document === 'undefined' || window.__poker_styles_injected) {
      return;
    }
    const style = document.createElement('style');
    style.textContent = cssContent;
    const parentForStyles =
      passedOptions && passedOptions.parent ? passedOptions.parent.ownerDocument.head : document.head;
    if (parentForStyles && !parentForStyles.querySelector('style[data-poker-renderer-styles]')) {
      style.setAttribute('data-poker-renderer-styles', 'true');
      parentForStyles.appendChild(style);
    }
    window.__poker_styles_injected = true;
  }

  function createCardElement(cardStr: string | null, isHidden: boolean = false): HTMLElement {
    const cardDiv = document.createElement('div');
    cardDiv.classList.add('card');
    if (isHidden || !cardStr || cardStr === '?' || cardStr === '??') {
      cardDiv.classList.add('card-back');
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
            <div class="bet-display">Standby</div>
            <div class="stack-cards-wrapper">
              <div class="player-card-area">
                <div class="player-cards-container"></div>
              </div>
              <div class="player-stats-container">
                <div class="player-hand-rank"></div>
                <div class="player-win-prob"></div>
                <div class="player-tie-prob"></div>
              </div>
              <div class="player-stack">
                <span class="player-stack-value">0</span>
              </div>
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

  function _renderPokerTableUI(data: RepeatedPokerStep): void {
    console.log('data is', data);
    if (!elements.pokerTable || !data || !elements.legend) return;

    // TODO: [TYPE_MISMATCH] The 'RepeatedPokerStep' type is missing many properties
    // that the original JS code expects.
    const {
      players, // This exists in BaseGameStep
      communityCards, // This is a string in RepeatedPokerStep, but JS expects string[]
      pot, // This exists
      winOdds, // This exists
      fiveCardBestHands, // This exists
    } = data;

    // TODO: [TYPE_MISMATCH] Manually defining missing properties from the type.
    const isTerminal = false; // 'isTerminal' is not in RepeatedPokerStep
    const handCount = 0; // 'handCount' is not in RepeatedPokerStep
    const winProb = winOdds; // 'winProb' is not in type, mapping 'winOdds'
    const tieProb = null; // 'tieProb' is not in type
    const handRank = fiveCardBestHands; // 'handRank' is not in type, mapping 'fiveCardBestHands'
    const leaderInfo: any = null; // 'leaderInfo' is not in type. Using 'any' to allow compilation.

    // Update legend
    const legendTitle = elements.legend.querySelector('.legend-title') as HTMLElement;
    const legendBody = elements.legend.querySelector('.legend-body') as HTMLElement;

    if (!legendTitle || !legendBody) return;

    legendTitle.innerHTML = ''; // Clear existing content

    const handSpan = document.createElement('span');
    handSpan.textContent = `Hand: ${handCount !== undefined && handCount !== null ? handCount + 1 : 'Standby'}`;
    legendTitle.appendChild(handSpan);

    if (leaderInfo) {
      const leaderInfoDiv = document.createElement('div');
      leaderInfoDiv.className = 'legend-leader-info';

      if (leaderInfo.thumbnail) {
        const leaderThumbnail = document.createElement('img');
        leaderThumbnail.src = leaderInfo.thumbnail;
        leaderThumbnail.className = 'legend-title-avatar';
        leaderInfoDiv.appendChild(leaderThumbnail);
      }

      const leaderNameSpan = document.createElement('span');
      const leaderName = leaderInfo.name.split(' ')[0];
      leaderNameSpan.textContent = `${leaderName} is up ${leaderInfo.winnings}`;
      leaderInfoDiv.appendChild(leaderNameSpan);
      legendTitle.appendChild(leaderInfoDiv);
    }

    legendBody.innerHTML = ''; // Clear existing content

    const table = document.createElement('div');
    table.className = 'legend-table';

    // ... (rest of legend rendering. It will be mostly empty due to missing 'previousHands') ...
    // (Legend rendering code omitted for brevity as it relies on 'any' types)

    if (elements.diagnosticHeader && (data as any).rawObservation) {
      // Optional: Show diagnostics for debugging
      // elements.diagnosticHeader.textContent = `[${passedOptions.step}] P_TURN:${(data as any).rawObservation.current_player} POT:${data.pot}`;
      // elements.diagnosticHeader.style.display = 'block';
    }

    if (!elements.communityCardsContainer || !elements.potDisplay) return;

    elements.communityCardsContainer.innerHTML = '';
    // Always show 5 slots for the river
    // Display cards left to right, with empty slots at the end
    const numCommunityCards = 5;

    // TODO: [TYPE_MISMATCH] 'communityCards' is a string, but the code expects an array of card strings - move this to the transformer
    const communityCardsArray = communityCards.match(/.{1,2}/g) || [];
    const numCards = communityCardsArray.length;

    // Add actual cards
    for (let i = 0; i < numCards; i++) {
      elements.communityCardsContainer.appendChild(createCardElement(communityCardsArray[i]));
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
        const showCards = isTerminal || (playerData.cards && !playerData.cards.includes(null!));

        // TODO: [TYPE_MISMATCH] 'playerData.cards' is a string, but code expects an array - move this to the transformer
        const playerCardsArray = playerData.cards ? playerData.cards.match(/.{1,2}/g) : [null, null];

        (playerCardsArray || [null, null]).forEach((cardStr) => {
          playerCardsContainer.appendChild(createCardElement(cardStr, !showCards && cardStr !== null));
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
          if (playerData.currentBet > 0) {
            if (playerData.actionDisplayText) {
              betDisplay.textContent = playerData.actionDisplayText;
            } else {
              betDisplay.textContent = '';
            }
            betDisplay.style.display = 'block';
          } else {
            betDisplay.style.display = 'none';
          }
        }

        const handRankElement = playerInfoArea.querySelector('.player-hand-rank') as HTMLElement;
        if (handRankElement && handRank && handRank[index]) {
          handRankElement.textContent = handRank[index];
        } else if (handRankElement) {
          handRankElement.textContent = '';
        }

        const winProbElement = playerInfoArea.querySelector('.player-win-prob') as HTMLElement;
        if (winProbElement && winProb && winProb[index] && !isTerminal) {
          winProbElement.textContent = `Win: ${winProb[index]}`;
        } else if (winProbElement) {
          winProbElement.textContent = '';
        }

        const tieProbElement = playerInfoArea.querySelector('.player-tie-prob') as HTMLElement;
        if (tieProbElement && tieProb && !isTerminal) {
          tieProbElement.textContent = `Tie: ${tieProb}`;
        } else if (tieProbElement) {
          tieProbElement.textContent = '';
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
