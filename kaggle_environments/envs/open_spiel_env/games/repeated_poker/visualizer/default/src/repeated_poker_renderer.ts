import poker_chip_1 from './images/poker_chip_1.svg';
import poker_chip_5 from './images/poker_chip_5.svg';
import poker_chip_10 from './images/poker_chip_10.svg';
import poker_chip_25 from './images/poker_chip_25.svg';
import poker_chip_100 from './images/poker_chip_100.svg';
import poker_card_back from './images/poker_card_back.svg';
import { RepeatedPokerStep, RepeatedPokerStepPlayer, LegacyRendererOptions } from '@kaggle-environments/core';
import { acpcCardToDisplay, calculateMatchStats, CardSuit, PlayerStats, suitSVGs } from './components/utils';
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  CategoryScale,
  Tooltip,
  Legend,
} from 'chart.js';

Chart.register(LineController, LineElement, PointElement, LinearScale, Title, CategoryScale, Tooltip, Legend);

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

export function renderer(options: LegacyRendererOptions): void {
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

    const legendBlinds = document.createElement('div');
    legendBlinds.className = 'legend-blinds';
    legendBlinds.textContent = 'Blinds: 1/2';
    elements.gameLayout.appendChild(legendBlinds);

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
    amount: number;
    winners: { name: string; thumbnail?: string }[];
    stepIndex: number;
  }

  interface DerivedLeaderboardInfo {
    topLeader: { name: string; winnings: number; thumbnail?: string } | null;
    completedHands: CompletedHand[];
  }

  function _deriveLeaderboardData(steps: RepeatedPokerStep[], currentStepIndex: number): DerivedLeaderboardInfo {
    const playerCumulativeRewards = new Map<number, number>();
    const playerDetails = new Map<number, { name: string; thumbnail?: string }>();
    const completedHands: CompletedHand[] = [];

    const relevantSteps = steps.slice(0, currentStepIndex + 1);

    // Map to store the LAST step of a hand (for results)
    const handsEndMap = new Map<number, RepeatedPokerStep>();
    // Map to store the FIRST array index of a hand (for navigation)
    const handsStartMap = new Map<number, number>();

    relevantSteps.forEach((step, index) => {
      // Track start index for this hand if we haven't seen it yet
      if (!handsStartMap.has(step.currentHandIndex)) {
        handsStartMap.set(step.currentHandIndex, index);
      }

      // Track end step (ignoring game-over to avoid double counting totals)
      if (step.stepType !== 'game-over') {
        handsEndMap.set(step.currentHandIndex, step);
      }
    });

    handsEndMap.forEach((lastStepOfHand, handIndex) => {
      const lastStepOfHandPlayers = lastStepOfHand.players as RepeatedPokerStepPlayer[];
      const isHandComplete = lastStepOfHandPlayers.some((p) => p.reward !== null);

      if (isHandComplete) {
        lastStepOfHandPlayers.forEach((p) => {
          if (!playerDetails.has(p.id)) {
            playerDetails.set(p.id, { name: p.name, thumbnail: p.thumbnail });
          }
          const currentTotal = playerCumulativeRewards.get(p.id) || 0;
          playerCumulativeRewards.set(p.id, currentTotal + (p.reward || 0));
        });

        const winners = lastStepOfHandPlayers.filter((p) => p.isWinner);
        if (winners.length > 0) {
          completedHands.push({
            handNum: handIndex + 1,
            amount: winners[0].reward || 0,
            winners: winners.map((w) => ({ name: w.name, thumbnail: w.thumbnail })),
            stepIndex: handsStartMap.get(handIndex) ?? 0, // Use the captured index
          });
        }
      }
    });

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

  function _renderLegendUI(
    steps: RepeatedPokerStep[],
    currentStepIndex: number,
    setCurrentStep: (step: number) => void
  ): void {
    if (!elements.legend || !steps || !steps[currentStepIndex]) return;

    const legendTitle = elements.legend.querySelector('.legend-title') as HTMLElement;
    const legendBody = elements.legend.querySelector('.legend-body') as HTMLElement;
    if (!legendTitle || !legendBody) return;

    const currentStepData = steps[currentStepIndex];
    const { currentHandIndex } = currentStepData;
    const { topLeader, completedHands } = _deriveLeaderboardData(steps, currentStepIndex);

    const allPlayerNames = new Set<string>();
    if (topLeader) allPlayerNames.add(topLeader.name);
    completedHands.forEach((h) => {
      h.winners.forEach((w) => allPlayerNames.add(w.name));
    });
    currentStepData.players.forEach((p) => allPlayerNames.add(p.name));

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
      completedHands
        .slice()
        .reverse()
        .forEach((hand) => {
          const row = document.createElement('div');
          row.className = 'legend-row';
          row.role = 'button';
          row.onclick = () => setCurrentStep(hand.stepIndex);

          const handCell = document.createElement('div');
          handCell.className = 'legend-cell';
          handCell.textContent = hand.handNum.toString();
          row.appendChild(handCell);

          const winnerCell = document.createElement('div');
          winnerCell.className = 'legend-cell';
          const winnerCellContainer = document.createElement('div');
          winnerCellContainer.className = 'legend-winner-cell';

          if (hand.winners.length > 1) {
            const splitSpan = document.createElement('span');
            splitSpan.textContent = 'Split Pot';
            splitSpan.classList.add('legend-split-pot');
            winnerCellContainer.appendChild(splitSpan);
          } else {
            const winner = hand.winners[0];
            if (winner.thumbnail) {
              const winnerThumbnail = document.createElement('img');
              winnerThumbnail.src = winner.thumbnail;
              winnerThumbnail.className = 'legend-avatar';
              winnerCellContainer.appendChild(winnerThumbnail);
            }
            const winnerNameSpan = document.createElement('span');
            winnerNameSpan.textContent = shortNameMap.get(winner.name) || winner.name;
            winnerCellContainer.appendChild(winnerNameSpan);
          }
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
      emptyCell.style.gridColumn = '1 / span 3';
      emptyRow.appendChild(emptyCell);
      table.appendChild(emptyRow);
    }
    legendBody.appendChild(table);
  }

  interface GraphPoint {
    hand: number;
    p0Total: number;
    p1Total: number;
  }

  function _extractGraphData(steps: RepeatedPokerStep[]): GraphPoint[] {
    const dataPoints: GraphPoint[] = [];

    // Start with 0 profit for both
    let p0Cumulative = 0;
    let p1Cumulative = 0;

    // Push initial state
    dataPoints.push({ hand: 0, p0Total: 0, p1Total: 0 });

    const handsMap = new Map<number, RepeatedPokerStep>();

    steps.forEach((step) => {
      // Ignore the 'game-over' step.
      // It contains the Total Cumulative Reward, which messes up our
      // step-by-step accumulation logic.
      if (step.stepType === 'game-over') return;
      handsMap.set(step.currentHandIndex, step);
    });

    // Iterate through hands in order
    const sortedHandIndices = Array.from(handsMap.keys()).sort((a, b) => a - b);

    sortedHandIndices.forEach((handIndex) => {
      const step = handsMap.get(handIndex);
      if (!step) return;

      const players = step.players as RepeatedPokerStepPlayer[];
      // Check if rewards exist for this hand
      const p0Reward = players.find((p) => p.id === 0)?.reward;
      const p1Reward = players.find((p) => p.id === 1)?.reward;

      // Only record data if the hand actually finished with rewards
      if (p0Reward !== null && p0Reward !== undefined) {
        p0Cumulative += p0Reward;
        p1Cumulative += p1Reward || 0;

        dataPoints.push({
          hand: handIndex + 1,
          p0Total: p0Cumulative,
          p1Total: p1Cumulative,
        });
      }
    });

    return dataPoints;
  }

  function _createTabNavigation(
    container: HTMLElement,
    tabs: string[],
    activeTab: string,
    onTabClick: (tab: string) => void
  ): void {
    const nav = document.createElement('div');
    nav.className = 'final-screen-tabs';

    tabs.forEach((tabName) => {
      const btn = document.createElement('button');
      btn.textContent = tabName;
      btn.className = `tab-button ${tabName === activeTab ? 'active' : ''}`;
      btn.onclick = () => onTabClick(tabName);
      nav.appendChild(btn);
    });

    container.appendChild(nav);
  }

  function _renderFinalScreenUI(currentStepData: RepeatedPokerStep): void {
    if (!elements.gameLayout) return;

    // 1. Clear Layout & Set Mode
    elements.gameLayout.innerHTML = '';
    elements.gameLayout.classList.add('final-screen-mode');

    // 2. Main Container
    const container = document.createElement('div');
    container.className = 'final-screen-container';

    // 3. Header / Winner Announcement
    let winner = (currentStepData.players as RepeatedPokerStepPlayer[]).find((p) => p.isWinner);
    if (!winner) {
      const players = currentStepData.players as RepeatedPokerStepPlayer[];
      winner = players.reduce((prev, current) => (prev.chipStack > current.chipStack ? prev : current));
    }

    const header = document.createElement('div');
    header.className = 'final-header';
    header.innerHTML = `
      <div class="final-title">Match Complete</div>
      <div class="final-subtitle">Winner: <span class="winner-text">${winner?.name || 'Player'}</span> (+${winner?.chipStack})</div>
    `;
    container.appendChild(header);

    // 4. Content Area
    const contentArea = document.createElement('div');
    contentArea.className = 'final-content-area';

    // State
    let currentTab = 'Graph';

    // 5. Render Function (Re-runs on Tab Switch)
    const renderTabs = () => {
      contentArea.innerHTML = '';

      // Render Nav
      _createTabNavigation(contentArea, ['Graph', 'History', 'Stats'], currentTab, (newTab) => {
        currentTab = newTab;
        renderTabs();
      });

      const body = document.createElement('div');
      body.className = 'tab-body';
      contentArea.appendChild(body);

      // --- TAB 1: GRAPH ---
      if (currentTab === 'Graph') {
        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'chart-container';
        const canvas = document.createElement('canvas');
        canvasContainer.appendChild(canvas);
        body.appendChild(canvasContainer);

        const graphData = _extractGraphData(options.steps as RepeatedPokerStep[]);

        new Chart(canvas, {
          type: 'line',
          data: {
            labels: graphData.map((d) => d.hand),
            datasets: [
              {
                label: currentStepData.players[0].name,
                data: graphData.map((d) => d.p0Total),
                backgroundColor: '#20BEFF',
                borderColor: '#20BEFF',
                pointStyle: 'circle',
                fill: true,
                tension: 0.1,
                pointRadius: 1,
              },
              {
                label: currentStepData.players[1].name,
                data: graphData.map((d) => d.p1Total),
                backgroundColor: '#F0510F',
                borderColor: '#F0510F',
                pointStyle: 'circle',
                fill: true,
                tension: 0.1,
                pointRadius: 1,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              title: { display: true, text: 'Cumulative Profit (Chips)', color: '#94a3b8' },
              legend: {
                labels: { usePointStyle: true, color: '#cbd5e1' },
              },
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'Hand',
                },
                ticks: { color: '#64748b' },
                grid: { color: '#334155' },
              },
              y: {
                title: {
                  display: true,
                  text: 'Chips',
                },
                ticks: { color: '#64748b' },
                grid: { color: '#334155' },
              },
            },
          },
        });
      }

      // --- TAB 2: HISTORY ---
      else if (currentTab === 'History') {
        const historyContainer = document.createElement('div');
        historyContainer.className = 'history-container';

        // 1. Generate Short Names Map
        const allNames = currentStepData.players.map((p) => p.name);
        const nameMap = _getDistinguishingNameMap(allNames);
        const p0Short = nameMap.get(currentStepData.players[0].name) || currentStepData.players[0].name;
        const p1Short = nameMap.get(currentStepData.players[1].name) || currentStepData.players[1].name;

        const hHeader = document.createElement('div');
        hHeader.className = 'history-row history-header';
        hHeader.innerHTML = `
          <div class="h-col h-num">#</div>
          <div class="h-col h-cards">${p0Short}</div>
          <div class="h-col h-board">Board</div>
          <div class="h-col h-cards">${p1Short}</div>
          <div class="h-col h-res">Winner</div>
        `;
        historyContainer.appendChild(hHeader);

        const { completedHands } = _deriveLeaderboardData(
          options.steps as RepeatedPokerStep[],
          options.steps.length - 1
        );
        const handsMap = new Map<number, RepeatedPokerStep>();
        (options.steps as RepeatedPokerStep[]).forEach((s) => {
          if (s.stepType !== 'game-over') handsMap.set(s.currentHandIndex, s);
        });

        completedHands
          .slice()
          .reverse()
          .forEach((h) => {
            const step = handsMap.get(h.handNum - 1);
            if (!step) return;

            const row = document.createElement('div');
            row.className = 'history-row clickable-row';
            row.onclick = () => options.setCurrentStep(h.stepIndex);

            // Result Logic
            const players = step.players as RepeatedPokerStepPlayer[];
            const winnerIndex = players.findIndex((p) => p.isWinner);
            const isSplit = players.filter((p) => p.isWinner).length > 1;

            // Calculate Winner Hand String for highlights
            // We simply check if the data exists. If it's a fold, this is likely empty, so no highlight occurs.
            let winningHandStr = '';
            if (!isSplit && step.bestFiveCardHands && step.bestFiveCardHands[winnerIndex]) {
              winningHandStr = step.bestFiveCardHands[winnerIndex];
            }

            const renderCards = (cardString: string | null) => {
              const container = document.createDocumentFragment();
              if (cardString) {
                const cards = cardString.match(/.{1,2}/g) || [];
                cards.forEach((c) => {
                  // Only highlight if card is part of the known winning hand string
                  const isWinningCard = !isSplit && winningHandStr.includes(c);
                  container.appendChild(createCardElement(c, false, isWinningCard));
                });
              }
              return container;
            };

            // 1. #
            const num = document.createElement('div');
            num.className = 'h-col h-num';
            num.textContent = String(h.handNum);
            row.appendChild(num);

            // 2. P0
            const p0Cards = document.createElement('div');
            p0Cards.className = 'h-col h-cards';
            if (!isSplit && winnerIndex !== 0) p0Cards.classList.add('dimmed-player');
            p0Cards.appendChild(renderCards((step.players[0] as RepeatedPokerStepPlayer).cards));
            row.appendChild(p0Cards);

            // 3. Board
            const board = document.createElement('div');
            board.className = 'h-col h-board';
            board.appendChild(renderCards(step.communityCards));
            row.appendChild(board);

            // 4. P1
            const p1Cards = document.createElement('div');
            p1Cards.className = 'h-col h-cards';
            if (!isSplit && winnerIndex !== 1) p1Cards.classList.add('dimmed-player');
            p1Cards.appendChild(renderCards((step.players[1] as RepeatedPokerStepPlayer).cards));
            row.appendChild(p1Cards);

            // 5. Result
            const res = document.createElement('div');
            res.className = 'h-col h-res';

            if (!isSplit) {
              const handWinner = players[winnerIndex];
              const winnerShortName = nameMap.get(handWinner.name) || handWinner.name;

              res.innerHTML = `
                <span class="res-name">${winnerShortName}</span>
                <span class="res-amt">+${handWinner.reward}</span>
            `;
              res.classList.add(handWinner.id === 0 ? 'win-p0' : 'win-p1');
            } else {
              res.innerHTML = `
                <span class="res-name">Split Pot</span>
                <span class="res-amt">${players[0].reward}</span>
             `;
            }
            row.appendChild(res);

            historyContainer.appendChild(row);
          });
        body.appendChild(historyContainer);
      }

      // --- TAB 3: STATS ---
      else if (currentTab === 'Stats') {
        const stats = calculateMatchStats(options.steps as RepeatedPokerStep[]);
        const statsContainer = document.createElement('div');
        statsContainer.className = 'stats-container';

        const createRow = (label: string, key: keyof PlayerStats, isPct = true, denKey?: keyof PlayerStats) => {
          const row = document.createElement('div');
          row.className = 'stat-row';
          row.innerHTML = `<div class="stat-label">${label}</div>`;

          // Helper to calc percent
          const getVal = (pStats: PlayerStats) => {
            if (!isPct) return String(pStats[key]);
            const totalHands = _deriveLeaderboardData(options.steps as RepeatedPokerStep[], options.steps.length - 1)
              .completedHands.length;
            const den = denKey ? pStats[denKey] : totalHands;
            return den > 0 ? `${((pStats[key] / den) * 100).toFixed(1)}%` : '0.0%';
          };

          const p0Val = document.createElement('div');
          p0Val.className = 'stat-val p0-stat';
          p0Val.textContent = getVal(stats.p0);

          const p1Val = document.createElement('div');
          p1Val.className = 'stat-val p1-stat';
          p1Val.textContent = getVal(stats.p1);

          // Reorder for: P0 - Label - P1
          row.innerHTML = '';
          row.appendChild(p0Val);
          const lbl = document.createElement('div');
          lbl.className = 'stat-label';
          lbl.textContent = label;
          row.appendChild(lbl);
          row.appendChild(p1Val);

          return row;
        };

        const sHeader = document.createElement('div');
        sHeader.className = 'stat-row stat-header';
        sHeader.innerHTML = `
           <div class="stat-val p0-stat">${currentStepData.players[0].name}</div>
           <div class="stat-label">METRIC</div>
           <div class="stat-val p1-stat">${currentStepData.players[1].name}</div>
        `;
        statsContainer.appendChild(sHeader);
        statsContainer.appendChild(createRow('VPIP', 'vpip'));
        statsContainer.appendChild(createRow('PFR', 'pfr'));
        statsContainer.appendChild(createRow('3-BET', 'threeBet'));
        statsContainer.appendChild(createRow('C-BET', 'cBetMade', true, 'cBetOps'));
        statsContainer.appendChild(createRow('FOLD TO C-BET', 'foldToCBet', true, 'faceCBetOps'));

        body.appendChild(statsContainer);
      }
    };

    renderTabs();
    container.appendChild(contentArea);
    elements.gameLayout.appendChild(container);
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

    // Check for split pot
    let isSplitPot = false;
    // If both players are winners, then it's a split pot
    if (stepType === 'final' && players.every((p) => (p as RepeatedPokerStepPlayer).isWinner)) {
      isSplitPot = true;
    }

    // Add actual cards
    for (let i = 0; i < numCards; i++) {
      const shouldHighlight = !isSplitPot && winnerBestHand.includes(communityCardsArray[i]);
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
      const playerData = basePlayerData as RepeatedPokerStepPlayer;

      const isWinner = playerData.actionDisplayText !== 'SPLIT POT' && playerData.isWinner;
      const isSplitPot = playerData.actionDisplayText === 'SPLIT POT';
      const isAllIn = playerData.actionDisplayText === 'ALL-IN';
      const isTurn = playerData.isTurn;

      const playerNameElement = elements.playerNames[index];
      if (playerNameElement) {
        playerNameElement.textContent = playerData.name;

        if (isTurn) {
          playerNameElement.classList.add('current-turn');
        } else {
          playerNameElement.classList.remove('current-turn');
        }

        // Add winner class if player won
        if (isWinner) {
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

        const shouldHighlightWinningHand = isWinner && showCards && isShowdown && bestHandArray.length > 0;
        (playerCardsArray || [null, null]).forEach((cardStr) => {
          const shouldHighlight = shouldHighlightWinningHand && cardStr && bestHandArray.includes(cardStr);
          playerCardsContainer.appendChild(
            createCardElement(cardStr, !showCards && cardStr !== null, !!shouldHighlight)
          );
        });
      }

      // Update chip stacks on the table
      if (elements.chipStacks[index]) {
        updateChipStack(elements.chipStacks[index], playerData.currentBetForStreet);
      }

      // Update info area (right side)
      const playerInfoArea = elements.playerInfoAreas[index];
      if (playerInfoArea) {
        // Highlight active player's pod
        if (isTurn) {
          playerInfoArea.classList.add('active-player');
        } else {
          playerInfoArea.classList.remove('active-player');
        }
        if (isSplitPot) {
          playerInfoArea.classList.add('split-pot');
        }

        // Highlight winner's pod
        if (isWinner) {
          playerInfoArea.classList.add('winner-player');
        } else {
          playerInfoArea.classList.remove('winner-player');
        }

        const stackValueEl = playerInfoArea.querySelector('.player-stack-value') as HTMLElement;
        if (stackValueEl) {
          stackValueEl.textContent = `Chips: ${playerData.chipStack}`;
        }

        const betDisplay = playerInfoArea.querySelector('.bet-display') as HTMLElement;
        if (betDisplay) {
          if (isWinner) {
            betDisplay.classList.add('winner-player');
          } else if (isSplitPot) {
            betDisplay.classList.add('split-pot');
          } else {
            if (isTurn) {
              betDisplay.classList.add('active-player');
            }
            if (isAllIn) {
              betDisplay.classList.add('all-in');
            }
          }
          if (playerData.currentBet > 0) {
            if (playerData.actionDisplayText) {
              betDisplay.textContent = playerData.actionDisplayText;
            } else {
              betDisplay.textContent = '';
            }
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
            minimumFractionDigits: 1,
          })}`;
          const tieOddsStringForPlayer = `TIE: ${winOdds[winOddsIndex + 1].toLocaleString(undefined, {
            style: 'percent',
            minimumFractionDigits: 1,
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
          const left = boxRect.left - containerRect.left - elements.dealerButton.offsetWidth;
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

  if (!_ensurePokerTableElements(parent)) {
    console.error('Renderer: Failed to ensure poker table elements.');
    parent.innerHTML = '<p style="color:red;">Error: Could not create poker table structure.</p>';
    return;
  }

  const currentStep = options.steps[options.step ?? 0] as RepeatedPokerStep;

  if (currentStep.stepType === 'game-over') {
    _renderFinalScreenUI(currentStep);
  } else {
    _renderPokerTableUI(currentStep);
    _renderLegendUI(options.steps as RepeatedPokerStep[], options.step ?? 0, options.setCurrentStep);
  }

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
