function _getLastMovesACPC(bettingString, currentPlayer) {
  // We will store all human-readable moves here
  const allMoves = [];

  // Split the action string by street (e.g., ["r5c", "cr11f"])
  const streets = bettingString.split('/');

  // Process each street's actions
  for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
    const streetAction = streets[streetIndex];
    let i = 0;

    // Preflop (streetIndex 0), action is "open" due to blinds.
    // Postflop (streetIndex > 0), action is "not open" (first player checks or bets).
    let isAggressiveActionOpen = (streetIndex === 0);

    // 4. Parse the moves within the street
    while (i < streetAction.length) {
      const char = streetAction[i];
      let move = null;

      if (char === 'c') {
        // 'c' (call/check)
        if (isAggressiveActionOpen) {
          move = 'call';
        } else {
          move = 'check';
        }
        isAggressiveActionOpen = false; // 'c' never leaves action open
        i++;
      } else if (char === 'f') {
        // 'f' (fold)
        move = 'fold';
        isAggressiveActionOpen = false; // 'f' ends the hand
        i++;
      } else if (char === 'r') {
        // 'r' (raise/bet)
        let amount = '';
        i++;
        // Continue to parse all digits of the raise amount
        while (i < streetAction.length && streetAction[i] >= '0' && streetAction[i] <= '9') {
          amount += streetAction[i];
          i++;
        }
        move = `raise ${amount}`;
        isAggressiveActionOpen = true; // 'r' always leaves action open
      } else {
        // Should not happen with valid input, but good to prevent infinite loops
        i++;
        continue;
      }

      // 5. Store this move in the history
      if (move) {
        allMoves.push(move);
      }
    }
  }

  // 6. Get the last two moves from our complete list
  const lastMove = allMoves.length > 0 ? allMoves[allMoves.length - 1] : null;
  const secondLastMove = allMoves.length > 1 ? allMoves[allMoves.length - 2] : null;

  const lastMoves = currentPlayer === 0 ? [secondLastMove, lastMove] : [lastMove, secondLastMove];

  return lastMoves;
}

function _parseStepHistoryData(universalPokerJSON) {
  const result = {
    cards: [],
    communityCards: '',
    bets: [],
    lastMoves: ['', ''],
    winOdds: [0, 0],
  };

  // Split the string into its main lines
  const lines = universalPokerJSON.acpc_state.trim().split('\n');
  if (lines.length < 2) {
    console.error("Invalid state string format.");
    return result;
  }

  const stateLine = lines[0]; // example: "STATE:0:r5c/cr11c/:6cKd|AsJc/7hQh6d/2c"
  const spentLine = lines[1]; // example: "Spent: [P0: 11  P1: 11  ]"

  // --- Parse the Spent Line ---
  if (spentLine) {
    const p0BetMatch = spentLine.match(/P0:\s*(\d+)/);
    const p1BetMatch = spentLine.match(/P1:\s*(\d+)/);

    const bets = [0, 0];

    if (p0BetMatch) {
      bets[0] = parseInt(p0BetMatch[1], 10);
    }

    if (p1BetMatch) {
      bets[1] = parseInt(p1BetMatch[1], 10);
    }

    result.bets = bets;
  }

  // --- Parse the State Line ---
  if (stateLine) {
    const stateParts = stateLine.split(':');

    // --- Parse Cards ---
    // The card string is always the last part
    const cardString = stateParts[stateParts.length - 1]; // example: "6cKd|AsJc/7hQh6d/2c"

    // Split card string by '/' to separate hand block from board blocks
    const cardSegments = cardString.split('/'); // example: ["6cKd|AsJc", "7hQh6d", "2c"]

    // Parse the first segment (player hands)
    if (cardSegments[0]) {
      const playerHands = cardSegments[0].split('|');
      if (playerHands.length >= 2) {
        // example: "6cKd"
        result.cards = [playerHands[0], playerHands[1]];
      }
    }

    // The rest of the segments are community cards, one per street
    result.communityCards = cardSegments
      .slice(1) // gets all elements AFTER the player hands
      .filter(Boolean) // removes any empty strings (e.g., from a trailing "/")
      .join(''); // joins the remaining segments into a single string

    // --- Parse Betting String --
    // The betting string is everything between the 2nd colon and the last colon.
    // This handles edge cases like "STATE:0:r5c/cr11c/:cards"
    const bettingString = stateParts.slice(2, stateParts.length - 1).join(':');

    if (bettingString) {
      result.lastMoves = _getLastMovesACPC(bettingString, universalPokerJSON.current_player);
    }
  }

  // Parse win odds
  const p0WinOdds = Number(universalPokerJSON.odds[0]).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 })
  const p1WinOdds = Number(universalPokerJSON.odds[1]).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 })
  result.winOdds = [p0WinOdds, p1WinOdds];

  return result;
}


function _getCurrentUniversalPokerFromStateHistory(stateHistory, step) {
  if (stateHistory) {

    const agentSteps = stateHistory.filter(s => JSON.parse(JSON.parse(s).current_universal_poker_json).current_player !== -1);
    const currentStep = agentSteps[step];
    return JSON.parse(JSON.parse(currentStep).current_universal_poker_json);
  }
  return null;
}


export const getPokerStateForStep = (environment, step) => {
  const numPlayers = 2;
  // --- Step Validation ---
  if (!environment || !environment.steps || !environment.steps[step] || !environment.info) {
    // return default state
    return null;
  }

  // --- Default State ---
  const stateUIData = {
    players: Array(numPlayers).fill(null).map((_, i) => {
      const agentName = environment?.info?.TeamNames?.[i] ||
        `Player ${i}`;
      const thumbnail = environment?.info?.Agents?.[i].ThumbnailUrl;
      return {
        id: `player${i}`,
        name: agentName,
        thumbnail: thumbnail,
        stack: 0,
        cards: [], // Will be filled with nulls or cards
        currentBet: 0,
        isDealer: i === 0,
        isTurn: false,
        reward: null
      };
    }),
    communityCards: [],
    pot: 0,
    isTerminal: false,
    blinds: [1, 2],
    lastMoves: [],
    rawObservation: null, // For debugging
    step: step,
    winOdds: [],
    fiveCardBestHands: [],
    lastMoves: [],
    currentPlayer: -1
  };

  // We have two sources for current game state: stepHistory and steps
  // This is because neither source contains all the information we need 

  const p0stateFromSteps = environment.steps[step][0];
  const p1stateFromSteps = environment.steps[step][1];

  const currentStateHistory = JSON.parse(environment.info.stateHistory[step]);
  const currentStateFromStateHistory = JSON.parse(currentStateHistory.current_universal_poker_json);

  // TODO: Handle the flop phase steps (chance steps)

  const currentUniversalPokerJSON = _getCurrentUniversalPokerFromStateHistory(environment.info.stateHistory, step);
  const currentStepFromStateHistory = _parseStepHistoryData(currentUniversalPokerJSON);

  const currentStepAgents = environment.steps[step];
  if (!currentStepAgents || currentStepAgents.length < numPlayers) {
    return stateUIData;
  }

  const pot_size = currentStepFromStateHistory.bets.reduce((a, b) => a + b, 0);
  const player_contributions = currentStepFromStateHistory.bets;
  const starting_stacks = currentUniversalPokerJSON.starting_stacks;
  const player_hands = [
    currentStepFromStateHistory.cards[0]?.match(/.{1,2}/g) || [],
    currentStepFromStateHistory.cards[1]?.match(/.{1,2}/g) || []
  ];
  const board_cards = currentStepFromStateHistory.communityCards ? currentStepFromStateHistory.communityCards.match(/.{1,2}/g).reverse() : [];

  // TODO: Add odds, best_five_card_hands best_hand_rank_types

  // TODO: Add current player

  const isTerminal = false // TODO: read isTerminal from observation
  stateUIData.isTerminal = isTerminal;
  stateUIData.pot = pot_size || 0;
  stateUIData.communityCards = board_cards || [];
  stateUIData.lastMoves = currentStepFromStateHistory.lastMoves;
  stateUIData.blinds = currentStateFromStateHistory.blinds;

  // --- Update Players ---
  for (let i = 0; i < numPlayers; i++) {
    const pData = stateUIData.players[i];
    const contribution = player_contributions ? player_contributions[i] : 0;
    const startStack = starting_stacks ? starting_stacks[i] : 0;

    pData.currentBet = contribution;
    pData.stack = startStack - contribution;
    pData.cards = (player_hands[i] || []).map(c => c === "??" ? null : c);
    pData.isTurn = p0stateFromSteps.observation.currentPlayer === i; // TODO: this may need to be flipped to show the other player responding to this move, which will display instantly
    pData.isDealer = currentStateFromStateHistory.blinds[i] === 1; // infer dealer from small blind

    if (isTerminal) {
      const reward = environment.rewards ? environment.rewards[i] : null;
      pData.reward = reward;
      if (reward > 0) {
        pData.name = `${pData.name} wins 🎉`;
        pData.isWinner = true;
        pData.status = null;
      } else {
        pData.status = null;
      }
    } else if (pData.stack === 0 && pData.currentBet > 0) {
      pData.status = "All-in";
    }
  }

  return stateUIData;
}