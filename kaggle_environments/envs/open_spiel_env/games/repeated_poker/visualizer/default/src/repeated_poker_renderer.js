import { getPokerStateForStep } from "./components/getRepeatedPokerStateForStep";
import { acpcCardToDisplay, suitSVGs } from "./components/utils";
import poker_chip_1 from "./images/poker_chip_1.svg";
import poker_chip_5 from "./images/poker_chip_5.svg";
import poker_chip_10 from "./images/poker_chip_10.svg";
import poker_chip_25 from "./images/poker_chip_25.svg";
import poker_chip_100 from "./images/poker_chip_100.svg";

export function renderer(options) {
  const chipImages = {
    1: poker_chip_1,
    5: poker_chip_5,
    10: poker_chip_10,
    25: poker_chip_25,
    100: poker_chip_100
  };

  const elements = {
    gameLayout: null,
    pokerTableContainer: null,
    pokerTable: null,
    communityCardsContainer: null,
    potDisplay: null,
    playersContainer: null,
    playerCardAreas: [],
    playerInfoAreas: [],
    playerThumbnails: [],
    dealerButton: null,
    chipStacks: [],
    diagnosticHeader: null,
    stepCounter: null
  };

  const css = `
    @font-face {
      font-family: 'Zeitung Pro';
      src:
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/l?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("woff2"),
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/d?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("woff"),
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/a?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("opentype");
        font-weight: normal;
        font-style: normal;
    }
    @font-face {
      font-family: 'Zeitung Pro';
      src:
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/l?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("woff2"),
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/d?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("woff"),
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/a?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("opentype");
        font-weight: bold;
        font-style: normal;
    }

    .poker-renderer-host {
      width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
      font-family: 'Zeitung Pro', sans-serif; background-color: #28303F; color: #fff;
      overflow: hidden; box-sizing: border-box; position: relative;
    }
    .poker-game-layout {
      width: 1000px;
      height: 900px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      transform-origin: center center;
    }
    .poker-table-container {
      width: 100%;
      height: 400px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .poker-table {
      width: 900px;
      height: 400px;
      background: radial-gradient(43.33% 50% at 50% 50%, #20BD48 0%, #0A4018 99.99%);
      border-radius: 300px;
      position: relative;
      border: 20px solid #5C3A21;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0;
      box-shadow: 0 8px 12px 6px rgba(0, 0, 0, 0.15), 0 4px 4px 0 rgba(0, 0, 0, 0.30);
    }
    .muck-line {
      position: absolute;
      width: 780px;
      height: 300px;
      border: 1px solid #9AA0A6;
      border-radius: 240px;
      pointer-events: none;
      z-index: 1;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .players-container {
      position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10;
    }
    .player-container {
      position: absolute;
      width: 100%;
      pointer-events: none;
      display: flex;
      flex-direction: column;
    }
    .player-container-0 { top: 0; }
    .player-container-1 { bottom: 0; flex-direction: column-reverse; }
    .player-card-area {
      color: white; text-align: center;
      display: flex; justify-content: left; align-items: left;
      pointer-events: auto;
      flex: 1;
    }
    .stack-cards-wrapper {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;
      gap: 16px;
      margin: 12px;
      width: 100%;
    }
    .player-info-area {
      color: white;
      width: auto;
      min-width: 200px;
      pointer-events: auto;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      margin: 20px auto;
      padding: 20px;
      background-color: rgba(32, 33, 36, 0.70);;
      border-radius: 16px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
      border: 2px solid transparent;
      transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .player-info-area.active-player {
      border-color: #20BEFF;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 20px rgba(32, 190, 255, 0.5);
    }
    .player-container-0 .player-info-area { flex-direction: column-reverse; }
    .player-name-wrapper {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 16px;
      margin: 0 60px;
      padding: 10px 0;
    }
    .player-thumbnail {
      width: 48px;
      height: 48px;
      border-radius: 50%;
      object-fit: cover;
      background-color: #ffffff;
      flex-shrink: 0;
      padding: 6px;
    }
    .player-name {
      font-size: 24px; font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: white;
      text-align: center;
    }
    .player-name.winner { color: #FFEB70; }
    .player-name.current-turn { color: #20BEFF; }
    .player-stack { font-size: 20px; font-weight: 600; color: #ffffff; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
    .player-cards-container { min-height: 80px; display: flex; justify-content: center; align-items:center;}
    .card {
      display: flex; flex-direction: column; justify-content: space-between; align-items: center;
      width: 44px; height: 70px; border: 2px solid #202124; border-radius: 8px;
      background-color: white; color: black; font-weight: bold; text-align: center; overflow: hidden; position: relative;
      padding: 6px;
      box-shadow: 0 6px 10px 4px rgba(0, 0, 0, 0.15), 0 2px 3px 0 rgba(0, 0, 0, 0.30);
    }
    .card-rank { font-family: 'Inter' sans-serif; font-size: 32px; line-height: 1; display: block; align-self: flex-start; }
    .card-suit { width: 36px; height: 36px; display: block; margin-bottom: 2px; }
    .player-cards-container .card { width: 38px; height: 60px; border-radius: 6px; }
    .player-cards-container .card:nth-child(2) { transform: rotate(20deg); margin-top: 14px; margin-left: -6px; }
    .player-cards-container .card-rank { font-size: 26px; }
    .player-cards-container .card-suit { width: 28px; height: 28px; }
    .card-suit svg { width: 100%; height: 100%; }
    .card-red .card-rank { color: #B3261E; }
    .card-red .card-suit svg { fill: #B3261E; }
    .card-black .card-rank { color: #000000; }
    .card-black .card-suit svg { fill: #000000; }
    .card-blue .card-rank { color: #0B57D0; }
    .card-blue .card-suit svg { fill: #0B57D0; }
    .card-green .card-rank { color: #146C2E; }
    .card-green .card-suit svg { fill: #146C2E; }
    .card-back {
      background-color: #2b6cb0;
      background-image: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%, rgba(255,255,255,0.1)),
                        linear-gradient(-45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%, rgba(255,255,255,0.1));
      background-size: 10px 10px; border: 2px solid #63b3ed;
    }
    .card-back .card-rank, .card-back .card-suit { display: none; }
    .card-empty {
      background-color: rgba(232, 234, 237, 0.1);
      border: 2px solid rgba(154, 160, 166, 0.5);
      box-shadow: none
    }
    .card-empty .card-rank, .card-empty .card-suit { display: none; }
    .community-cards-area { text-align: center; z-index: 10; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
    .community-cards-container { min-height: 75px; display: flex; justify-content: center; align-items:center; margin-bottom: 0.5rem; gap: 8px; }
    .pot-display { font-size: 30px; font-weight: bold; color: #ffffff; margin-bottom: 10px; }
    .bet-display {
      display: inline-block; padding: 10px 20px; border-radius: 30px;
      background-color: #ffffff; color: black;
      font-family: 'Inter' sans-serif; font-size: 20px; font-weight: 600;
      text-align: center;
      height: 20pxrem; line-height: 20px;
      width: 150px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .blind-indicator { font-size: 0.7rem; color: #a0aec0; margin-top: 3px; }
    .dealer-button {
      width: 36px; height: 36px; background-color: #f0f0f0; color: #333; border-radius: 50%;
      text-align: center; font-weight: bold; font-size: 28px; position: absolute;
      padding-left: 1px;
      line-height: 33px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3); z-index: 15; pointer-events: auto;
      border: 2px solid black;
      outline: 2px solid #20BEFF;
      left: 320px
    }
    .dealer-button.dealer-player0 { top: 170px; }
    .dealer-button.dealer-player1 { bottom: 170px; }
    .step-counter {
      position: absolute; top: 12px; right: 12px; z-index: 20;
      background-color: rgba(60, 64, 67, 0.9); color: #ffffff;
      padding: 6px 12px; border-radius: 6px;
      font-size: 14px; font-weight: 600;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .chip-stack {
      position: absolute;
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 12px;
      z-index: 12;
      pointer-events: none;
      left: 50%;
      transform: translateX(-50%);
    }
    .chip-stack.chip-stack-player0 {
      top: 60px;
    }
    .chip-stack.chip-stack-player1 {
      bottom: 60px;
    }
    .chip-stack-chips {
      display: flex;
      flex-direction: row-reverse;
      align-items: flex-end;
      justify-content: center;
      gap: 8px;
      position: relative;
    }
    .chip-denomination-stack {
      display: flex;
      flex-direction: column-reverse;
      align-items: center;
      position: relative;
    }
    .chip {
      width: 40px;
      height: 40px;
      position: relative;
      margin-bottom: -34px;
      filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
    }
    .chip:first-child {
      margin-bottom: 0;
    }
    .chip img {
      width: 100%;
      height: 100%;
      display: block;
    }
    .chip-stack-label {
      color: #FFFFFF;
      font-size: 18px;
      font-weight: bold;
      white-space: nowrap;
    }
  `;

  function _injectStyles(passedOptions) {
    if (typeof document === 'undefined' || window.__poker_styles_injected) {
      return;
    }
    const style = document.createElement('style');
    style.textContent = css;
    const parentForStyles =
      passedOptions && passedOptions.parent ? passedOptions.parent.ownerDocument.head : document.head;
    if (parentForStyles && !parentForStyles.querySelector('style[data-poker-renderer-styles]')) {
      style.setAttribute('data-poker-renderer-styles', 'true');
      parentForStyles.appendChild(style);
    }
    window.__poker_styles_injected = true;
  }

  function createCardElement(cardStr, isHidden = false) {
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

      if (suitSVGs[suit]) {
        suitSpan.innerHTML = suitSVGs[suit];
      }

      cardDiv.appendChild(suitSpan);

      if (suit === 'hearts') cardDiv.classList.add('card-red');
      else if (suit === 'spades') cardDiv.classList.add('card-black');
      else if (suit === 'diamonds') cardDiv.classList.add('card-blue');
      else if (suit === 'clubs') cardDiv.classList.add('card-green');
    }
    return cardDiv;
  }

  function updateChipStack(chipStackElement, betAmount) {
    if (betAmount <= 0) {
      chipStackElement.style.display = 'none';
      return;
    }

    chipStackElement.style.display = 'flex';
    const chipsContainer = chipStackElement.querySelector('.chip-stack-chips');
    const labelElement = chipStackElement.querySelector('.chip-stack-label');

    chipsContainer.innerHTML = '';
    labelElement.textContent = betAmount;

    // Break down bet into denominations (100, 25, 10, 5, 1)
    const denominations = [100, 25, 10, 5, 1];
    let remaining = betAmount;
    const chipCounts = [];

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
  function _ensurePokerTableElements(parentElement, passedOptions) {
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
              <div class="player-stack">
                <span class="player-stack-value">0</span>
              </div>
            </div>
            `;
      playerContainer.appendChild(playerInfoArea);
      elements.playerInfoAreas.push(playerInfoArea);

      // Get reference to card area (already in DOM)
      const playerCardArea = playerInfoArea.querySelector('.player-card-area');
      elements.playerCardAreas.push(playerCardArea);
    }

    elements.dealerButton = document.createElement('div');
    elements.dealerButton.className = 'dealer-button';
    elements.dealerButton.textContent = 'D';
    elements.dealerButton.style.display = 'none';
    elements.playersContainer.appendChild(elements.dealerButton);

    elements.stepCounter = document.createElement('div');
    elements.stepCounter.className = 'step-counter';
    elements.stepCounter.textContent = 'Standby';
    elements.gameLayout.appendChild(elements.stepCounter);
    return true;
  }

  // --- State Parsing ---
  function _parseKagglePokerState(options) {
    const { environment, step } = options;
    const numPlayers = 2;

    // --- Default State ---
    const defaultStateUiData = {
      players: [],
      communityCards: [],
      pot: 0,
      isTerminal: false,
    };

    // --- Step Validation ---
    if (!environment || !environment.steps || !environment.steps[step] || !environment.info?.stateHistory) {
      return defaultStateUiData;
    }

    return getPokerStateForStep(environment, step);
  }

  function _applyScale(parentElement) {
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

  function _renderPokerTableUI(data, passedOptions) {
    if (!elements.pokerTable || !data) return;
    const { players, communityCards, pot, isTerminal, step } = data;

    // Update step counter
    if (elements.stepCounter && step !== undefined) {
      elements.stepCounter.textContent = `Step: ${step}`;
    }

    if (elements.diagnosticHeader && data.rawObservation) {
      // Optional: Show diagnostics for debugging
      // elements.diagnosticHeader.textContent = `[${passedOptions.step}] P_TURN:${data.rawObservation.current_player} POT:${data.pot}`;
      // elements.diagnosticHeader.style.display = 'block';
    }

    elements.communityCardsContainer.innerHTML = '';
    // Always show 5 slots for the river
    // Display cards left to right, with empty slots at the end
    const numCommunityCards = 5;
    const numCards = communityCards ? communityCards.length : 0;

    // Since the 4th and 5th street cards are appended to the communityCards array, we need to
    // reverse it so that the added cards are put at the end of the display area on the board.
    if (communityCards) communityCards.reverse();

    // Add actual cards
    for (let i = 0; i < numCards; i++) {
      elements.communityCardsContainer.appendChild(createCardElement(communityCards[i]));
    }

    // Fill remaining slots with empty cards
    for (let i = numCards; i < numCommunityCards; i++) {
      const emptyCard = document.createElement('div');
      emptyCard.classList.add('card', 'card-empty');
      elements.communityCardsContainer.appendChild(emptyCard);
    }

    elements.potDisplay.textContent = `Total Pot : ${pot}`;

    players.forEach((playerData, index) => {
      const playerNameElement = elements.playerNames[index];
      if (playerNameElement) {
        const playerNameText =
          playerData.isTurn && !isTerminal ? `${playerData.name} responding...` : playerData.name;
        playerNameElement.textContent = playerNameText;

        // Highlight current player's turn
        if (playerData.isTurn && !isTerminal) {
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
        playerThumbnailElement.src = playerData.thumbnail;
        playerThumbnailElement.style.display = 'block';
      } else if (playerThumbnailElement) {
        playerThumbnailElement.style.display = 'none';
      }

      // Update card area (left side)
      const playerCardArea = elements.playerCardAreas[index];
      if (playerCardArea) {
        const playerCardsContainer = playerCardArea.querySelector('.player-cards-container');
        playerCardsContainer.innerHTML = '';

        // In heads-up, we show both hands at the end.
        const showCards = isTerminal || (playerData.cards && !playerData.cards.includes(null));

        (playerData.cards || [null, null]).forEach((cardStr) => {
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
        if (playerData.isTurn && !isTerminal) {
          playerInfoArea.classList.add('active-player');
        } else {
          playerInfoArea.classList.remove('active-player');
        }

        playerInfoArea.querySelector('.player-stack-value').textContent = `${playerData.stack}`;

        const betDisplay = playerInfoArea.querySelector('.bet-display');
        if (playerData.currentBet > 0) {
          if (data.lastMoves[index]) {
            betDisplay.textContent = data.lastMoves[index];
          } else {
            if (playerData.isDealer) {
              betDisplay.textContent = 'small blind';
            } else {
              betDisplay.textContent = 'big blind';
            }
          }
          betDisplay.style.display = 'block';
        } else {
          betDisplay.style.display = 'none';
        }
      }
    });

    const dealerPlayerIndex = players.findIndex((p) => p.isDealer);
    if (elements.dealerButton) {
      if (dealerPlayerIndex !== -1) {
        elements.dealerButton.style.display = 'block';
        // Remove previous dealer class
        elements.dealerButton.classList.remove('dealer-player0', 'dealer-player1');
        // Add new dealer class based on player index
        elements.dealerButton.classList.add(`dealer-player${dealerPlayerIndex}`);
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

  if (!_ensurePokerTableElements(parent, options)) {
    console.error('Renderer: Failed to ensure poker table elements.');
    parent.innerHTML = '<p style="color:red;">Error: Could not create poker table structure.</p>';
    return;
  }

  const uiData = _parseKagglePokerState(options);
  _renderPokerTableUI(uiData, options);

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
