function _getActionStringsFromACPC(bettingString, nextPlayerIndex, numPlayers = 2) {
  const moves = [];

  // Split the action string by street (e.g., ["r5c", "cr11f"])
  const streets = bettingString.split('/');

  // Process each street's actions
  for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
    const streetAction = streets[streetIndex];
    let i = 0;

    // 4. Parse the moves within the street
    while (i < streetAction.length) {
      const char = streetAction[i];

      if (char === 'r') {
        let amount = '';
        i++;
        // parse all digits of the raise amount
        while (i < streetAction.length && streetAction[i] >= '0' && streetAction[i] <= '9') {
          amount += streetAction[i];
          i++;
        }
        moves.push(`r${amount}`)
      } else {
        moves.push(char);
        i++;
        continue;
      }
    }
  }

  // 6. Get the last two moves from our complete list
  const lastMove = moves.length > 0 ? moves[moves.length - 1] : null;

  const actionStrings = Array(numPlayers).fill('');

  if (lastMove) {
    if (typeof nextPlayerIndex === 'number' && nextPlayerIndex >= 0) {
      const lastActor = (nextPlayerIndex + numPlayers - 1) % numPlayers;
      actionStrings[lastActor] = lastMove;
    } else {
      const inferredActor = (moves.length - 1) % numPlayers;
      actionStrings[inferredActor] = lastMove;
    }
  }

  return actionStrings;
}

function _parseStepHistoryData(universalPokerJSON, nextPlayerIndex, numPlayers = 2) {
  const result = {
    cards: [],
    communityCards: '',
    bets: [],
    playerActionStrings: Array(numPlayers).fill(''),
    winOdds: [0, 0],
  };

  if (!universalPokerJSON) {
    return result;
  }

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
      result.playerActionStrings = _getActionStringsFromACPC(
        bettingString,
        nextPlayerIndex,
        numPlayers
      );
    }
  }

  // Parse win odds
  const p0WinOdds = Number(universalPokerJSON.odds[0]).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 })
  const p1WinOdds = Number(universalPokerJSON.odds[1]).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 })
  result.winOdds = [p0WinOdds, p1WinOdds];

  return result;
}
export const getPokerStateForStep = (environment, step) => {
  const numPlayers = 2;
  // --- Step Validation ---
  if (!environment || !environment.steps || !environment.steps[step] || !environment.info) {
    // return default state
    return null;
  }

  const stepsWithEndStates = environment.steps;

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
        reward: null,
        actionDisplayText: ""
      };
    }),
    communityCards: [],
    pot: 0,
    isTerminal: false,
    rawObservation: null, // For debugging
    step: step,
    winOdds: [],
    fiveCardBestHands: [],
    currentPlayer: -1,
    winner: -1,
  };

  // We have two sources for current game state: stepHistory and steps
  // This is because neither source contains all the information we need 
  const currentStepData = stepsWithEndStates[step];

  let currentPlayer = null;

  const currentStateHistoryEntry = currentStepData?.stateHistory
    ? JSON.parse(currentStepData.stateHistory)
    : null;
  const currentStateFromStateHistory = currentStateHistoryEntry
    ? JSON.parse(currentStateHistoryEntry.current_universal_poker_json)
    : null;

  const observation = currentStepData?.step?.observation;
  if (typeof observation?.currentPlayer === "number") {
    currentPlayer = observation.currentPlayer;
  } else if (typeof observation?.current_player === "number") {
    currentPlayer = observation.current_player;
  }

  const currentUniversalPokerJSON = currentStateFromStateHistory;
  const currentStepFromStateHistory = _parseStepHistoryData(
    currentUniversalPokerJSON,
    currentStateFromStateHistory?.current_player,
    numPlayers
  );

  const actionString = currentStepData?.step?.action?.actionString || "";
  let actionPlayerIndex = null;
  let actionDisplayText = "";
  if (actionString) {
    const playerMatch = actionString.match(/player=(\d+)/);
    const moveMatch = actionString.match(/move=([^\s]+)/);
    if (playerMatch) {
      actionPlayerIndex = parseInt(playerMatch[1], 10);
      if (moveMatch) {
        const moveRaw = moveMatch[1];
        const moveLower = moveRaw.toLowerCase();
        if (moveLower.startsWith("bet")) {
          const amountMatch = moveRaw.match(/\d+/);
          actionDisplayText = amountMatch ? `r${amountMatch[0]}` : "bet";
        } else if (moveLower.startsWith("raise")) {
          const amountMatch = moveRaw.match(/\d+/);
          actionDisplayText = amountMatch ? `r${amountMatch[0]}` : "r";
        } else if (moveLower === "call") {
          actionDisplayText = "c";
        } else if (moveLower === "check") {
          actionDisplayText = "k";
        } else if (moveLower === "fold") {
          actionDisplayText = "f";
        } else {
          actionDisplayText = moveRaw;
        }
      }
    }
  }

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
  stateUIData.currentPlayer = currentPlayer ?? -1;

  // --- Update Players ---
  for (let i = 0; i < numPlayers; i++) {
    const pData = stateUIData.players[i];
    const contribution = player_contributions ? player_contributions[i] : 0;
    const startStack = starting_stacks ? starting_stacks[i] : 0;

    pData.currentBet = contribution;
    pData.stack = startStack - contribution;
    pData.cards = (player_hands[i] || []).map(c => c === "??" ? null : c);
    pData.isTurn = currentPlayer === i;
    const dealerIndex = typeof currentStateHistoryEntry?.dealer === "number"
      ? currentStateHistoryEntry.dealer
      : 0;
    pData.isDealer = dealerIndex === i;
    pData.actionDisplayText = actionPlayerIndex === i ? actionDisplayText : "";

    if (currentStepData.isEndState) {
      pData.isTurn = false;
      pData.isWinner = currentStepData.winner === i;
      if (currentStepData.winner === i) {
        pData.actionDisplayText = "WINNER"
      } else {
        if (currentStepData.handConclusion = "fold") {
          pData.actionDisplayText = "FOLD"
        } else {
          pData.actionDisplayText = "LOSER"
        }
      }
    }

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
