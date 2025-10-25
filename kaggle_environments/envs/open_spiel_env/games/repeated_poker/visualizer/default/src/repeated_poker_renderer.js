import { getPokerStateForStep } from "./components/getRepeatedPokerStateForStep";
import { acpcCardToDisplay, suitSVGs } from "./components/utils";

export function renderer(options) {
    const elements = {
        gameLayout: null,
        pokerTableContainer: null,
        pokerTable: null,
        communityCardsContainer: null,
        potDisplay: null,
        playersContainer: null,
        playerCardAreas: [],
        playerInfoAreas: [],
        dealerButton: null,
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
      font-family: 'Zeitung Pro', sans-serif; background-color: #1C1D20; color: #fff;
      overflow: hidden; padding: 1rem; box-sizing: border-box; position: relative;
    }
    .poker-game-layout { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; position: relative; max-width: 750px; max-height: 750px; }
    .poker-table-container { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; max-width: 750px; max-height: 275px; }
    .poker-table {
      width: clamp(400px, 85vw, 750px); height: clamp(220px, 48vw, 275px);
      background-color: #197631; border-radius: 24px; position: relative;
      display: flex; align-items: center; justify-content: center;
      margin: 0 60px;
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
    .player-container-0 { bottom: 0; flex-direction: column-reverse; }
    .player-container-1 { top: 0; }
    .player-area-wrapper {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .player-card-area {
      margin: 20px 60px; color: white; text-align: center;
      display: flex; flex-direction: column; justify-content: center; align-items: center;
      min-height: 100px; pointer-events: auto;
    }
    .player-info-area {
      color: white;
      min-width: 180px;
      pointer-events: auto;
      display: flex;
      flex-direction: column;
      justify-content: left;
      align-items: left;
      margin-right: 60px;
    }
    .player-container-0 .player-info-area { flex-direction: column-reverse; }
    .player-name {
      font-size: 32px; font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: white;
      text-align: left;
      padding: 10px 0;
      margin: 0 60px;
    }
    .player-name.winner { color: #FFEB70; }
    .player-stack { font-size: 32px; font-weight: 600; color: #ffffff; margin: 16px 0; display: flex; justify-content: space-between; align-items: center; }
    .player-cards-container { min-height: 70px; display: flex; justify-content: flex-start; align-items:center; gap: 12px; }
    .card {
      display: flex; flex-direction: column; justify-content: space-between; align-items: center;
      width: 80px; height: 112px; border: 2px solid #202124; border-radius: 8px;
      background-color: white; color: black; font-weight: bold; text-align: center; overflow: hidden; position: relative;
      padding: 6px;
    }
    .card-rank { font-family: 'Inter' sans-serif; font-size: 50px; line-height: 1; display: block; align-self: flex-start; }
    .card-suit { width: 50px; height: 50px; display: block; margin-bottom: 2px; }
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
      background-color: rgba(255, 255, 255, 0.1);
      border: 2px solid rgba(32, 33, 36, 0.5);
      background-image: none;
    }
    .card-empty .card-rank, .card-empty .card-suit { display: none; }
    .community-cards-area { text-align: center; z-index: 10; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
    .community-cards-container { min-height: 75px; display: flex; justify-content: center; align-items:center; margin-bottom: 0.5rem; gap: 12px; }
    .pot-display { font-size: 40px; font-weight: bold; color: #ffffff; margin-bottom: 30px; }
    .bet-display {
      display: inline-block; padding: 10px 20px; border-radius: 12px;
      background-color: #3C4043; color: #ffff;
      font-family: 'Inter' sans-serif; font-size: 1.75rem; font-weigth: 600;
      text-align: center;
      height: 3rem; line-height: 3rem;
      min-width: 200px;
    }
    .blind-indicator { font-size: 0.7rem; color: #a0aec0; margin-top: 3px; }
    .dealer-button {
      width: 36px; height: 36px; background-color: #f0f0f0; color: #333; border-radius: 50%;
      text-align: center; line-height: 36px; font-weight: bold; font-size: 1.5rem; position: absolute;
      border: 3px solid #1EBEFF; box-shadow: 0 1px 3px rgba(0,0,0,0.3); z-index: 15; pointer-events: auto;
    }
    .dealer-button.dealer-player0 { bottom: 110px; }
    .dealer-button.dealer-player1 { top: 110px; }
    .step-counter {
      position: absolute; top: 12px; right: 12px; z-index: 20;
      background-color: rgba(60, 64, 67, 0.9); color: #ffffff;
      padding: 6px 12px; border-radius: 6px;
      font-size: 14px; font-weight: 600;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    @media (max-width: 768px) {
      .bet-display { font-size: 1.5rem; height: 2.2rem; line-height: 2.2rem; min-width: 0;}
      .card { width: 60px; height: 85px; } .card-rank { font-size: 35px; } .card-suit { width: 35px; height: 35px; }
      .community-cards-container { gap: 6px; }
      .player-card-area { min-height: 120px; }
      .player-cards-container { gap: 6px; }
      .player-info-area { min-width: 160px; }
      .poker-game-layout { max-height: 700px; }
      .pot-display { font-size: 35px; margin-bottom: 20px; }
    }
    @media (max-width: 600px) {
      .bet-display { font-size: 20px; height: 40px; line-height: 40px; }
      .card { width: 50px; height: 70px; padding: 2px; } .card-rank { font-size: 32px; } .card-suit { width: 32px; height: 32px; }
      .community-cards-container { gap: 2px; }
      .dealer-button { font-size: 20px; height: 24px; line-height: 24px; width: 24px; }
      .dealer-button.dealer-player0 { bottom: 95px; }
      .dealer-button.dealer-player1 { top: 95px; }
      .player-card-area { min-height: 110px; margin: 0 0 0 40px;}
      .player-cards-container { gap: 2px; }
      .player-info-area { margin-right: 20px; }
      .player-name { font-size: 30px; margin: 0 20px; }
      .player-stack { font-size: 30px; }
      .poker-game-layout { max-height: 600px; }
      .poker-table { width: clamp(300px, 90vw, 600px); height: clamp(160px, 50vw, 200px); margin: 20px; }
      .pot-display { font-size: 30px; margin-bottom: 20px; }
    }
    @media (max-width: 400px) {
      .bet-display { font-size: 15px; height: 30px; line-height: 30px; }
      .card { width: 40px; height: 56px; margin: 0 2px; padding: 2px; } .card-rank { font-size: 25px; } .card-suit { width: 25px; height: 25px; }
      .community-cards-container { gap: 2px; }
      .dealer-button { font-size: 15px; height: 20px; line-height: 20px; width: 20px; }
      .dealer-button.dealer-player0 { bottom: 85px; }
      .dealer-button.dealer-player1 { top: 85px; }
      .player-card-area { margin: 0 0 0 30px;}
      .player-cards-container { gap: 2px; }
      .player-info-area { min-width: 100px; margin-right: 0; }
      .player-name { font-size: 25px; }
      .player-stack { font-size: 15px; }
      .poker-game-layout { max-height: 500px; }
      .poker-table { width: clamp(280px, 95vw, 380px); height: clamp(150px, 55vw, 150px); margin: 0;}
      .pot-display { font-size: 25px; margin-bottom: 15px; }
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

        for (let i = 0; i < 2; i++) {
            // Create player container that groups all player elements
            const playerContainer = document.createElement('div');
            playerContainer.className = `player-container player-container-${i}`;
            elements.playersContainer.appendChild(playerContainer);
            elements.playerContainers.push(playerContainer);

            // Player name
            const playerName = document.createElement('div');
            playerName.className = `player-name`;
            playerName.textContent = `Player ${i}`;
            playerContainer.appendChild(playerName);
            elements.playerNames.push(playerName);

            // Create wrapper for card and info areas
            const playerAreaWrapper = document.createElement('div');
            playerAreaWrapper.className = 'player-area-wrapper';
            playerContainer.appendChild(playerAreaWrapper);

            // Card area (left side)
            const playerCardArea = document.createElement('div');
            playerCardArea.className = `player-card-area`;
            playerCardArea.innerHTML = `
        <div class="player-cards-container"></div>
      `;
            playerAreaWrapper.appendChild(playerCardArea);
            elements.playerCardAreas.push(playerCardArea);

            // TODO: Render chip stack
            // Info area (right side)
            const playerInfoArea = document.createElement('div');
            playerInfoArea.className = `player-info-area`;
            playerInfoArea.innerHTML = `
        <div class="player-stack">
            <span class="player-stack-value">0</span>
        </div>
        <div class="bet-display" style="display:none;">Bet : 0</div>
      `;
            playerAreaWrapper.appendChild(playerInfoArea);
            elements.playerInfoAreas.push(playerInfoArea);
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

        elements.potDisplay.textContent = `Pot : ${pot}`;

        players.forEach((playerData, index) => {
            const playerNameElement = elements.playerNames[index];
            if (playerNameElement) {
                const playerNameText =
                    playerData.isTurn && !isTerminal ? `${playerData.name} responding...` : playerData.name;
                playerNameElement.textContent = playerNameText;

                // Add winner class if player won
                if (playerData.isWinner) {
                    playerNameElement.classList.add('winner');
                } else {
                    playerNameElement.classList.remove('winner');
                }
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

            // Update info area (right side)
            const playerInfoArea = elements.playerInfoAreas[index];
            if (playerInfoArea) {
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
}
