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
            display: flex; flex-direction: column; justify-content: space-between; align-items: center;
            width: 48px; height: 72px; border: 1px solid #202124; border-radius: 8px; margin: 0 3px;
            background-color: white; color: black; font-weight: bold; text-align: center; overflow: hidden; position: relative;
            padding: 4px;
        }
        .card-rank { font-size: 2.1rem; line-height: 1; display: block; align-self: flex-start; }
        .card-suit { width: 33px; height: 33px; display: block; margin-bottom: 2px; }
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
        .community-cards-area { text-align: center; z-index: 10; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
        .community-cards-container { min-height: 75px; display: flex; justify-content: center; align-items:center; margin-bottom: 0.5rem; }
        .community-cards-container .card { width: 52px; height: 76px; }
        .community-cards-container .card-rank { font-size: 2.4rem; } .community-cards-container .card-suit { width: 39px; height: 39px; }
        .pot-display { font-size: 1.1rem; font-weight: bold; color: #ffffff; }
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
            .card { width: 44px; height: 66px; } .card-rank { font-size: 1.95rem; } .card-suit { width: 30px; height: 30px; }
            .community-cards-container .card { width: 48px; height: 70px; }
            .community-cards-container .card-rank { font-size: 2.25rem;} .community-cards-container .card-suit { width: 36px; height: 36px; }
        }
        @media (max-width: 600px) {
            .poker-table { width: clamp(300px, 90vw, 500px); height: clamp(160px, 50vw, 250px); border-radius: 125px; }
            .player-pod { min-width: 100px; max-width: 140px; padding: 0.4rem 0.5rem; font-size: 0.85rem; min-height: 110px;}
            .player-name { font-size: 0.85rem;} .player-stack { font-size: 0.75rem; }
            .card { width: 40px; height: 60px; margin: 0 2px; } .card-rank { font-size: 1.8rem; } .card-suit { width: 27px; height: 27px; }
            .community-cards-container .card { width: 42px; height: 62px; }
            .community-cards-container .card-rank { font-size: 1.95rem;} .community-cards-container .card-suit { width: 30px; height: 30px; }
            .bet-display { font-size: 0.75rem; } .pos-player0-sb { bottom: -45px; } .pos-player1-bb { top: -45px; }
            .dealer-button { width: 32px; height: 32px; line-height: 32px; font-size: 0.9rem;}
            .dealer-sb { bottom: -8px; left: calc(50% + 75px); }
        }
        @media (max-width: 400px) {
            .poker-table { width: clamp(280px, 95vw, 380px); height: clamp(150px, 55vw, 200px); border-radius: 100px; border-width: 8px; }
            .player-pod { min-width: 90px; max-width: 120px; padding: 0.3rem 0.4rem; min-height: 100px;}
            .player-name { font-size: 0.8rem;} .player-stack { font-size: 0.7rem; }
            .card { width: 36px; height: 54px; margin: 0 1px; } .card-rank { font-size: 1.65rem; } .card-suit { width: 24px; height: 24px; }
            .community-cards-container .card { width: 38px; height: 56px; }
            .community-cards-container .card-rank { font-size: 1.8rem;} .community-cards-container .card-suit { width: 27px; height: 27px; }
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

    const suitSVGs = {
        spades: '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><path d="M31.9017 8.5459L31.9027 8.54688V8.54785L31.9642 8.62988C32.0049 8.68463 32.0647 8.76567 32.1419 8.86914C32.2968 9.07661 32.5214 9.37584 32.7982 9.74316C33.3519 10.4781 34.1164 11.4847 34.9583 12.5713C36.6487 14.7529 38.6314 17.23 39.8587 18.4951C40.5956 19.2546 42.6938 21.1061 45.0882 23.3057C47.4623 25.4866 50.1062 27.9917 51.8763 30.0146C53.659 32.052 54.5809 34.6512 54.9242 37.0439C55.2443 39.2762 55.07 41.3963 54.5648 42.7754L54.4593 43.041L54.4583 43.0439C54.2366 43.5604 53.4581 45.3752 51.889 47.0635C50.312 48.7602 47.9209 50.3437 44.5003 50.3438C41.1459 50.3437 38.4383 49.3111 36.5716 48.2812C35.668 47.7827 34.959 47.2827 34.4662 46.8984C34.6764 47.5682 35.0067 48.3733 35.5287 49.2432C35.8458 49.7716 36.3961 50.2525 37.0941 50.6953C37.7874 51.1352 38.5874 51.513 39.3636 51.8545C40.1218 52.1881 40.8886 52.4987 41.4437 52.7803C41.7223 52.9216 41.9834 53.0734 42.181 53.2383C42.3602 53.3878 42.5999 53.6413 42.5999 54C42.5999 54.3241 42.4172 54.5729 42.2318 54.7422C42.0426 54.9148 41.7911 55.0617 41.5101 55.1895C40.9443 55.4466 40.1512 55.6745 39.1976 55.8652C37.2827 56.2482 34.6237 56.5 31.5999 56.5C28.5752 56.5 25.9176 56.2484 23.9427 55.8662C22.957 55.6754 22.1264 55.4487 21.5003 55.1982C21.1878 55.0732 20.9125 54.9375 20.6908 54.7881C20.4761 54.6434 20.2705 54.4592 20.1527 54.2236L20.0999 54.1182V54C20.0999 53.6414 20.3397 53.3878 20.5189 53.2383C20.7165 53.0734 20.9776 52.9216 21.2562 52.7803C21.8113 52.4987 22.578 52.1881 23.3363 51.8545C24.1124 51.513 24.9125 51.1352 25.6058 50.6953C26.3038 50.2525 26.8541 49.7716 27.1712 49.2432C27.726 48.3186 28.0632 47.467 28.2708 46.7734C28.2466 46.7955 28.2233 46.8199 28.1976 46.8428C27.7563 47.2352 27.101 47.7542 26.2376 48.2725C24.5092 49.3098 21.9429 50.3437 18.5863 50.3438C15.1655 50.3437 12.7737 48.7603 11.1966 47.0635C9.6273 45.3749 8.84884 43.56 8.62728 43.0439L8.62631 43.041C8.04128 41.6783 7.81998 39.4248 8.16146 37.0439C8.50467 34.6513 9.42677 32.052 11.2093 30.0146C12.9793 27.9918 15.6234 25.4865 17.9977 23.3057C20.3921 21.1061 22.4903 19.2546 23.2272 18.4951C24.4545 17.23 26.4372 14.7529 28.1276 12.5713C28.9695 11.4847 29.734 10.4781 30.2877 9.74316C30.5645 9.37584 30.7891 9.07661 30.944 8.86914C31.0212 8.76567 31.081 8.68463 31.1217 8.62988L31.1832 8.54785V8.54688L31.1842 8.5459C31.4355 8.19531 31.8496 8 32.2859 8H32.7999C33.2363 8 33.6504 8.19531 33.9017 8.5459Z"/></svg>',
        hearts: '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><path d="M31.6667 56.9333L27.8 53.4667C23.3111 49.4222 19.6 45.9333 16.6667 43C13.7333 40.0667 11.4 37.4333 9.66667 35.1C7.93333 32.7667 6.72222 30.6222 6.03333 28.6667C5.34444 26.7111 5 24.7111 5 22.6667C5 18.4889 6.4 15 9.2 12.2C12 9.4 15.4889 8 19.6667 8C21.9778 8 24.1778 8.48889 26.2667 9.46667C28.3556 10.4444 30.1556 11.8222 31.6667 13.6C33.1778 11.8222 34.9778 10.4444 37.0667 9.46667C39.1556 8.48889 41.3556 8 43.6667 8C47.8444 8 51.3333 9.4 54.1333 12.2C56.9333 15 58.3333 18.4889 58.3333 22.6667C58.3333 24.7111 57.9889 26.7111 57.3 28.6667C56.6111 30.6222 55.4 32.7667 53.6667 35.1C51.9333 37.4333 49.6 40.0667 46.6667 43C43.7333 45.9333 40.0222 49.4222 35.5333 53.4667L31.6667 56.9333Z"/></svg>',
        diamonds: '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><path d="M32 58.3333L8 31.6667L32 5L56 31.6667L32 58.3333Z"/></svg>',
        clubs: '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><path d="M32.7422 8C39.0131 8.00014 44.0965 13.0836 44.0967 19.3545C44.0967 22.3905 42.9028 25.1463 40.9619 27.1836C42.108 26.7945 43.3357 26.5811 44.6133 26.5811C50.8842 26.5813 55.9678 31.6646 55.9678 37.9355C55.9677 44.2065 50.8842 49.2898 44.6133 49.29C40.7767 49.29 37.3866 47.3859 35.3311 44.4727C35.3545 44.6869 35.4 44.9939 35.4873 45.3721C35.6708 46.1669 36.0397 47.2784 36.7832 48.5176C37.1124 49.0661 37.683 49.5639 38.4043 50.0215C39.121 50.4762 39.9477 50.8671 40.749 51.2197C41.5324 51.5644 42.323 51.8854 42.8955 52.1758C43.1826 52.3214 43.4509 52.4767 43.6533 52.6455C43.8375 52.7992 44.0801 53.0572 44.0801 53.4199C44.0799 53.7476 43.8956 54.0007 43.7061 54.1738C43.5126 54.3503 43.2539 54.5014 42.9648 54.6328C42.3825 54.8974 41.5654 55.1324 40.582 55.3291C38.6066 55.7241 35.8618 55.9844 32.7412 55.9844C29.6198 55.9843 26.8772 55.7244 24.8398 55.3301C23.8233 55.1333 22.9671 54.9005 22.3223 54.6426C22.0002 54.5137 21.7169 54.3731 21.4893 54.2197C21.2688 54.0712 21.0593 53.8831 20.9395 53.6436L20.8867 53.5381V53.4199C20.8867 53.0575 21.1294 52.7992 21.3135 52.6455C21.5159 52.4766 21.7851 52.3214 22.0723 52.1758C22.6447 51.8855 23.4346 51.5643 24.2178 51.2197C25.019 50.8672 25.8458 50.4761 26.5625 50.0215C27.2837 49.5639 27.8543 49.066 28.1836 48.5176C28.9271 47.2784 29.297 46.1669 29.4805 45.3721C29.5675 44.9951 29.6113 44.6888 29.6348 44.4746C27.579 47.3866 24.1901 49.29 20.3545 49.29C14.0836 49.2899 9.00003 44.2065 9 37.9355C9 31.6646 14.0835 26.5812 20.3545 26.5811C21.9457 26.5811 23.4603 26.9091 24.835 27.5C22.7097 25.4365 21.3867 22.5506 21.3867 19.3545C21.3869 13.0835 26.4712 8 32.7422 8Z"/></svg>'
    }

    function acpcCardToDisplay(acpcCard) {
        if (!acpcCard || acpcCard.length < 2) return { rank: '?', suit: '', original: acpcCard };
        const rankChar = acpcCard[0].toUpperCase();
        const suitChar = acpcCard[1].toLowerCase();
        const rankMap = { 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A' };
        const suitMap = { 's': 'spades', 'h': 'hearts', 'd': 'diamonds', 'c': 'clubs' };
        const rank = rankMap[rankChar] || rankChar;
        const suit = suitMap[suitChar] || '';
        return { rank, suit, original: acpcCard };
    }

    function createCardElement(cardStr, isHidden = false) {
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        if (isHidden || !cardStr || cardStr === '?' || cardStr === "??") {
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
            obsP0 = JSON.parse(currentStepAgents[0].observation.observationString);
            obsP1 = JSON.parse(currentStepAgents[1].observation.observationString);
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

        elements.potDisplay.textContent = `Pot : ${pot}`;

        players.forEach((playerData, index) => {
            const playerPod = elements.playerPods[index];
            if (!playerPod) return;

            playerPod.querySelector('.player-name').textContent = playerData.name;
            playerPod.querySelector('.player-stack').textContent = `${playerData.stack}`;

            const playerCardsContainer = playerPod.querySelector('.player-cards-container');
            playerCardsContainer.innerHTML = '';

            // In heads-up, we show both hands at the end.
            const showCards = isTerminal || (playerData.cards && !playerData.cards.includes(null));

            (playerData.cards || [null, null]).forEach(cardStr => {
                playerCardsContainer.appendChild(createCardElement(cardStr, !showCards && cardStr !== null));
            });

            const betDisplay = playerPod.querySelector('.bet-display');
            if (playerData.currentBet > 0) {
                betDisplay.textContent = `${playerData.currentBet}`;
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