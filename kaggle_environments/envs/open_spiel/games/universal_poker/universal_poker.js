function renderer(options) {
    // --- Existing Elements and Style Injection (Unchanged) ---
    const elements = {
        pokerTableContainer: null,
        pokerTable: null,
        communityCardsContainer: null,
        potDisplay: null,
        playerPods: [],
        dealerButton: null,
        diagnosticHeader: null,
        gameMessageArea: null,
    };

    function _injectStyles(passedOptions) {
        if (typeof document === 'undefined' || window.__poker_styles_injected) {
            return;
        }
        const style = document.createElement('style');
        style.textContent = `
        .poker-renderer-host {
            width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
            font-family: 'Inter', sans-serif; background-color: #2d3748; color: #fff;
            overflow: hidden; padding: 1rem; box-sizing: border-box;
        }
        .poker-table-container { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
        .poker-table {
            width: clamp(400px, 85vw, 850px); height: clamp(220px, 48vw, 450px);
            background-color: #006400; border-radius: 225px; position: relative;
            border: 12px solid #5c3a21; box-shadow: 0 0 25px rgba(0,0,0,0.6);
            display: flex; align-items: center; justify-content: center;
        }
        .player-pod {
            background-color: rgba(0, 0, 0, 0.75); border: 1px solid #4a5568; border-radius: 0.75rem;
            padding: 0.6rem 0.8rem; color: white; text-align: center; position: absolute;
            min-width: 120px; max-width: 160px; box-shadow: 0 3px 12px rgba(0,0,0,0.35);
            transform: translateX(-50%); display: flex; flex-direction: column; justify-content: space-between;
            min-height: 130px;
        }
        .player-name { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; margin-bottom: 0.25rem; font-size: 0.9rem;}
        .player-stack { font-size: 0.8rem; color: #facc15; margin-bottom: 0.25rem; }
        .player-cards-container { margin: 0.25rem 0; min-height: 70px; display: flex; justify-content: center; align-items:center;}
        .player-status { font-size: 0.75rem; color: #9ca3af; min-height: 1.1em; margin-top: 0.25rem; }
        .card {
            display: inline-flex; flex-direction: column; justify-content: center; align-items: center;
            width: 48px; height: 68px; border: 1px solid #999; border-radius: 0.375rem; margin: 0 3px;
            background-color: white; color: black; font-weight: bold; text-align: center; overflow: hidden; position: relative;
        }
        .card-rank { font-size: 1.8rem; line-height: 1; display: block; margin-top: 2px; }
        .card-suit { font-size: 1.5rem; line-height: 1; display: block; }
        .card-red .card-suit { color: #c0392b; } .card-black .card-suit { color: #1a202c; }
        .card-back {
            background-color: #2b6cb0;
            background-image: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%, rgba(255,255,255,0.1)),
                                linear-gradient(-45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%, rgba(255,255,255,0.1));
            background-size: 10px 10px; border: 2px solid #63b3ed;
        }
        .card-back .card-rank, .card-back .card-suit { display: none; }
        .community-cards-area { text-align: center; z-index: 10; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
        .community-cards-container { min-height: 75px; display: flex; justify-content: center; align-items:center; margin-bottom: 0.5rem; }
        .community-cards-container .card { width: 52px; height: 72px; }
        .community-cards-container .card-rank { font-size: 2rem; } .community-cards-container .card-suit { font-size: 1.7rem; }
        .pot-display { font-size: 1.1rem; font-weight: bold; color: #facc15; }
        .bet-display {
            display: inline-block; min-width: 55px; padding: 4px 8px; border-radius: 12px;
            background-color: #1a202c; color: #f1c40f; font-size: 0.8rem; line-height: 1.4;
            text-align: center; margin-top: 4px; border: 1.5px solid #f1c40f; box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        .blind-indicator { font-size: 0.7rem; color: #a0aec0; margin-top: 3px; }
        .dealer-button {
            width: 36px; height: 36px; background-color: #f0f0f0; color: #333; border-radius: 50%;
            text-align: center; line-height: 36px; font-weight: bold; font-size: 1rem; position: absolute;
            border: 2px solid #888; box-shadow: 0 1px 3px rgba(0,0,0,0.3); z-index: 5;
        }
        .pos-player0-sb { bottom: -55px; left: 50%; }
        .pos-player1-bb { top: -55px; left: 50%; }
        .dealer-sb { bottom: -15px; left: calc(50% + 95px); transform: translateX(-50%); }
        .current-player-turn-highlight { border: 2px solid #f1c40f !important; box-shadow: 0 0 15px #f1c40f, 0 3px 12px rgba(0,0,0,0.35) !important; }
        #game-message-area { position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background-color: rgba(0,0,0,0.6); padding: 5px 10px; border-radius: 5px; font-size: 0.9rem; z-index: 20;}

        @media (max-width: 768px) {
            .poker-table { width: clamp(350px, 90vw, 700px); height: clamp(180px, 48vw, 350px); border-radius: 175px; }
            .pos-player0-sb { bottom: -50px; } .pos-player1-bb { top: -50px; }
            .dealer-sb { left: calc(50% + 85px); bottom: -12px; }
            .player-pod { min-width: 110px; max-width: 150px; padding: 0.5rem 0.7rem; min-height: 120px; }
            .card { width: 44px; height: 62px; } .card-rank { font-size: 1.6rem; } .card-suit { font-size: 1.3rem; }
            .community-cards-container .card { width: 48px; height: 68px; }
            .community-cards-container .card-rank { font-size: 1.8rem;} .community-cards-container .card-suit { font-size: 1.5rem;}
        }
        @media (max-width: 600px) {
            .poker-table { width: clamp(300px, 90vw, 500px); height: clamp(160px, 50vw, 250px); border-radius: 125px; }
            .player-pod { min-width: 100px; max-width: 140px; padding: 0.4rem 0.5rem; font-size: 0.85rem; min-height: 110px;}
            .player-name { font-size: 0.85rem;} .player-stack { font-size: 0.75rem; }
            .card { width: 40px; height: 58px; margin: 0 2px; } .card-rank { font-size: 1.4rem; } .card-suit { font-size: 1.2rem; }
            .community-cards-container .card { width: 42px; height: 60px; }
            .community-cards-container .card-rank { font-size: 1.5rem;} .community-cards-container .card-suit { font-size: 1.3rem;}
            .bet-display { font-size: 0.75rem; } .pos-player0-sb { bottom: -45px; } .pos-player1-bb { top: -45px; }
            .dealer-button { width: 32px; height: 32px; line-height: 32px; font-size: 0.9rem;}
            .dealer-sb { bottom: -8px; left: calc(50% + 75px); }
        }
        @media (max-width: 400px) {
            .poker-table { width: clamp(280px, 95vw, 380px); height: clamp(150px, 55vw, 200px); border-radius: 100px; border-width: 8px; }
            .player-pod { min-width: 90px; max-width: 120px; padding: 0.3rem 0.4rem; min-height: 100px;}
            .player-name { font-size: 0.8rem;} .player-stack { font-size: 0.7rem; }
            .card { width: 36px; height: 52px; margin: 0 1px; } .card-rank { font-size: 1.2rem; } .card-suit { font-size: 1rem; }
            .community-cards-container .card { width: 38px; height: 55px; }
            .community-cards-container .card-rank { font-size: 1.3rem;} .community-cards-container .card-suit { font-size: 1.1rem;}
            .dealer-button { width: 28px; height: 28px; line-height: 28px; font-size: 0.8rem;}
            .pos-player0-sb { bottom: -40px; } .pos-player1-bb { top: -40px; }
            .dealer-sb { bottom: -5px; left: calc(50% + 65px); }
        }
        `;
        const parentForStyles = passedOptions && passedOptions.parent ? passedOptions.parent.ownerDocument.head : document.head;
        if (parentForStyles && !parentForStyles.querySelector('style[data-poker-renderer-styles]')) {
            style.setAttribute('data-poker-renderer-styles', 'true');
            parentForStyles.appendChild(style);
        }
        window.__poker_styles_injected = true;
    }

    function acpcCardToDisplay(acpcCard) {
        if (!acpcCard || acpcCard.length < 2) return { rank: '?', suitSymbol: '', original: acpcCard };
        const rankChar = acpcCard[0].toUpperCase();
        const suitChar = acpcCard[1].toLowerCase();
        const rankMap = { 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A' };
        const suitMap = { 's': '♠', 'h': '♥', 'd': '♦', 'c': '♣' };
        const rank = rankMap[rankChar] || rankChar;
        const suitSymbol = suitMap[suitChar] || '';
        return { rank, suitSymbol, original: acpcCard };
    }

    function createCardElement(cardStr, isHidden = false) {
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        if (isHidden || !cardStr || cardStr === '?' || cardStr === "??") {
            cardDiv.classList.add('card-back');
        } else {
            const { rank, suitSymbol } = acpcCardToDisplay(cardStr);
            const rankSpan = document.createElement('span');
            rankSpan.classList.add('card-rank');
            rankSpan.textContent = rank;
            cardDiv.appendChild(rankSpan);
            const suitSpan = document.createElement('span');
            suitSpan.classList.add('card-suit');
            suitSpan.textContent = suitSymbol;
            cardDiv.appendChild(suitSpan);
            if (suitSymbol === '♥' || suitSymbol === '♦') cardDiv.classList.add('card-red');
            else if (suitSymbol === '♠' || suitSymbol === '♣') cardDiv.classList.add('card-black');
        }
        return cardDiv;
    }

    function _ensurePokerTableElements(parentElement, passedOptions) {
        if (!parentElement) return false;
        parentElement.innerHTML = '';
        parentElement.classList.add('poker-renderer-host');

        elements.diagnosticHeader = document.createElement('h1');
        elements.diagnosticHeader.id = 'poker-renderer-diagnostic-header';
        elements.diagnosticHeader.textContent = "Poker Table Initialized (Live Data)";
        elements.diagnosticHeader.style.cssText = "color: lime; background-color: black; padding: 5px; font-size: 12px; position: absolute; top: 0px; left: 0px; z-index: 10001; display: none;"; // Hidden by default
        parentElement.appendChild(elements.diagnosticHeader);

        elements.gameMessageArea = document.createElement('div');
        elements.gameMessageArea.id = 'game-message-area';
        parentElement.appendChild(elements.gameMessageArea);

        elements.pokerTableContainer = document.createElement('div');
        elements.pokerTableContainer.className = 'poker-table-container';
        parentElement.appendChild(elements.pokerTableContainer);

        elements.pokerTable = document.createElement('div');
        elements.pokerTable.className = 'poker-table';
        elements.pokerTableContainer.appendChild(elements.pokerTable);

        const communityArea = document.createElement('div');
        communityArea.className = 'community-cards-area';
        elements.pokerTable.appendChild(communityArea);

        elements.communityCardsContainer = document.createElement('div');
        elements.communityCardsContainer.className = 'community-cards-container';
        communityArea.appendChild(elements.communityCardsContainer);

        elements.potDisplay = document.createElement('div');
        elements.potDisplay.className = 'pot-display';
        communityArea.appendChild(elements.potDisplay);

        elements.playerPods = [];
        for (let i = 0; i < 2; i++) {
            const playerPod = document.createElement('div');
            playerPod.className = `player-pod ${i === 0 ? 'pos-player0-sb' : 'pos-player1-bb'}`;
            playerPod.innerHTML = `
                <div class="player-name">Player ${i}</div>
                <div class="player-stack">$0.00</div>
                <div class="player-cards-container"></div>
                <div class="bet-display" style="display:none;">$0.00</div>
                <div class="player-status">(${i === 0 ? 'SB' : 'BB'})</div>
            `;
            elements.pokerTable.appendChild(playerPod);
            elements.playerPods.push(playerPod);
        }

        elements.dealerButton = document.createElement('div');
        elements.dealerButton.className = 'dealer-button dealer-sb';
        elements.dealerButton.textContent = 'D';
        elements.dealerButton.style.display = 'none';
        elements.pokerTable.appendChild(elements.dealerButton);
        return true;
    }


    // --- REVISED PARSING LOGIC ---
    function _parseKagglePokerState(options) {
        const { environment, step } = options;
        const numPlayers = 2; // Assuming 2 players based on logs

        // --- Default State ---
        const defaultUIData = {
            players: Array(numPlayers).fill(null).map((_, i) => ({
                id: `player${i}`,
                name: `Player ${i}`,
                stack: 0,
                cards: [], // Will be filled with nulls or cards
                currentBet: 0,
                position: i === 0 ? "SB" : "BB",
                isDealer: i === 0,
                isTurn: false,
                status: "Waiting...",
                reward: null
            })),
            communityCards: [],
            pot: 0,
            isTerminal: false,
            gameMessage: "Initializing...",
            rawObservation: null, // For debugging
        };

        // --- Step Validation ---
        if (!environment || !environment.steps || !environment.steps[step]) {
            return defaultUIData;
        }
        const currentStepAgents = environment.steps[step];
        if (!currentStepAgents || currentStepAgents.length < numPlayers) {
            defaultUIData.gameMessage = "Waiting for agent data...";
            return defaultUIData;
        }

        // --- Observation Extraction & Merging ---
        let obsP0 = null, obsP1 = null;
        try {
            obsP0 = JSON.parse(currentStepAgents[0].observation.observation_string);
            obsP1 = JSON.parse(currentStepAgents[1].observation.observation_string);
        } catch (e) {
            defaultUIData.gameMessage = "Error parsing observation JSON.";
            return defaultUIData;
        }

        if (!obsP0) {
            defaultUIData.gameMessage = "Waiting for valid game state...";
            return defaultUIData;
        }

        // --- Combine observations into a single, reliable state object ---
        const combinedState = { ...obsP0 }; // Start with Player 0's data
        // Player hands are split across observations. We need to merge them.
        combinedState.player_hands = [
             // Take the real hand from P0's obs
            obsP0.player_hands[0].length > 0 ? obsP0.player_hands[0] : [],
            // Take the real hand from P1's obs
            obsP1.player_hands[1].length > 0 ? obsP1.player_hands[1] : []
        ];

        defaultUIData.rawObservation = combinedState;

        // --- Populate UI Data from Combined State ---
        const {
            pot_size,
            player_contributions,
            starting_stacks,
            player_hands,
            board_cards,
            current_player,
            betting_history,
        } = combinedState;

        const isTerminal = current_player === "terminal";
        defaultUIData.isTerminal = isTerminal;
        defaultUIData.pot = pot_size || 0;
        defaultUIData.communityCards = board_cards || [];


        // --- Update Player Pods ---
        for (let i = 0; i < numPlayers; i++) {
            const pData = defaultUIData.players[i];
            const contribution = player_contributions ? player_contributions[i] : 0;
            const startStack = starting_stacks ? starting_stacks[i] : 0;

            pData.currentBet = contribution;
            pData.stack = startStack - contribution;
            pData.cards = (player_hands[i] || []).map(c => c === "??" ? null : c);
            pData.isTurn = String(i) === String(current_player);
            pData.status = pData.position; // Default status

            if (isTerminal) {
                 const reward = environment.rewards ? environment.rewards[i] : null;
                 pData.reward = reward;
                 if (reward > 0) pData.status = "Winner!";
                 else if (reward < 0) pData.status = "Loser";
                 else pData.status = "Game Over";
            } else if (pData.isTurn) {
                pData.status = "Thinking...";
            } else if (pData.stack === 0 && pData.currentBet > 0) {
                pData.status = "All-in";
            }
        }
        
        // Handle folded player status
        if (!isTerminal && betting_history && betting_history.includes('f')) {
            // A simple fold check: the player who didn't make the last action and isn't the current player might have folded.
            // This is a simplification. A more robust parser would track the betting sequence.
            const lastAction = betting_history.slice(-1);
            if (lastAction === 'f') {
                // Find who is NOT the current player
                const nonCurrentPlayerIndex = current_player === '0' ? 1 : 0;
                // If they are not all-in, they folded.
                if (defaultUIData.players[nonCurrentPlayerIndex].status !== 'All-in') {
                    defaultUIData.players[nonCurrentPlayerIndex].status = "Folded";
                }
            }
        }


        // --- Set Game Message ---
        if (isTerminal) {
            const winnerIndex = environment.rewards ? environment.rewards.findIndex(r => r > 0) : -1;
            if (winnerIndex !== -1) {
                defaultUIData.gameMessage = `Player ${winnerIndex} wins!`;
            } else {
                 defaultUIData.gameMessage = "Game Over.";
            }
        } else if (current_player === "chance") {
             defaultUIData.gameMessage = `Dealing...`;
        } else {
            defaultUIData.gameMessage = `Player ${current_player}'s turn.`;
        }

        return defaultUIData;
    }


    // --- RENDERER UI LOGIC (Unchanged) ---
    function _renderPokerTableUI(data, passedOptions) {
        if (!elements.pokerTable || !data) return;
        const { players, communityCards, pot, isTerminal, gameMessage } = data;

        if (elements.diagnosticHeader && data.rawObservation) {
            // Optional: Show diagnostics for debugging
            // elements.diagnosticHeader.textContent = `[${passedOptions.step}] P_TURN:${data.rawObservation.current_player} POT:${data.pot}`;
            // elements.diagnosticHeader.style.display = 'block';
        }
        if (elements.gameMessageArea) {
            elements.gameMessageArea.textContent = gameMessage;
        }

        elements.communityCardsContainer.innerHTML = '';
        if (communityCards && communityCards.length > 0) {
            communityCards.forEach(cardStr => {
                elements.communityCardsContainer.appendChild(createCardElement(cardStr));
            });
        }

        elements.potDisplay.textContent = `Pot: $${pot}`;

        players.forEach((playerData, index) => {
            const playerPod = elements.playerPods[index];
            if (!playerPod) return;

            playerPod.querySelector('.player-name').textContent = playerData.name;
            playerPod.querySelector('.player-stack').textContent = `$${playerData.stack}`;

            const playerCardsContainer = playerPod.querySelector('.player-cards-container');
            playerCardsContainer.innerHTML = '';

            // In heads-up, we show both hands at the end.
            const showCards = isTerminal || (playerData.cards && !playerData.cards.includes(null));

            (playerData.cards || [null, null]).forEach(cardStr => {
                playerCardsContainer.appendChild(createCardElement(cardStr, !showCards && cardStr !== null));
            });

            const betDisplay = playerPod.querySelector('.bet-display');
            if (playerData.currentBet > 0) {
                betDisplay.textContent = `$${playerData.currentBet}`;
                betDisplay.style.display = 'inline-block';
            } else {
                betDisplay.style.display = 'none';
            }

            playerPod.querySelector('.player-status').textContent = playerData.status;

            if (playerData.isTurn && !isTerminal) {
                playerPod.classList.add('current-player-turn-highlight');
            } else {
                playerPod.classList.remove('current-player-turn-highlight');
            }
        });

        const dealerPlayer = players.find(p => p.isDealer);
        if (elements.dealerButton) {
            elements.dealerButton.style.display = dealerPlayer ? 'block' : 'none';
        }
    }

    // --- MAIN EXECUTION LOGIC ---
    const { parent } = options;
    if (!parent) {
        console.error("Renderer: Parent element not provided.");
        return;
    }

    _injectStyles(options);

    if (!_ensurePokerTableElements(parent, options)) {
        console.error("Renderer: Failed to ensure poker table elements.");
        parent.innerHTML = '<p style="color:red;">Error: Could not create poker table structure.</p>';
        return;
    }

    // Use the revised parsing logic
    const uiData = _parseKagglePokerState(options);
    _renderPokerTableUI(uiData, options);
}