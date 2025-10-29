const PLACEHOLDER_CARD = '2c';

function _getActionStringsFromACPC(bettingString, nextPlayerIndex, numPlayers = 2) {
  const moves = [];
  const streets = bettingString.split('/');
  for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
    const streetAction = streets[streetIndex];
    let i = 0;
    while (i < streetAction.length) {
      const char = streetAction[i];
      if (char === 'r') {
        let amount = '';
        i++;
        while (i < streetAction.length && streetAction[i] >= '0' && streetAction[i] <= '9') {
          amount += streetAction[i];
          i++;
        }
        moves.push(`r${amount}`);
      } else {
        moves.push(char);
        i++;
      }
    }
  }

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

  const lines = universalPokerJSON.acpc_state.trim().split('\n');
  if (lines.length < 2) {
    return result;
  }

  const stateLine = lines[0];
  const spentLine = lines[1];

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

  if (stateLine) {
    const stateParts = stateLine.split(':');
    const cardString = stateParts[stateParts.length - 1];
    const cardSegments = cardString.split('/');

    if (cardSegments[0]) {
      const playerHands = cardSegments[0].split('|');
      if (playerHands.length >= 2) {
        result.cards = [playerHands[0], playerHands[1]];
      }
    }

    result.communityCards = cardSegments
      .slice(1)
      .filter(Boolean)
      .join('');

    const bettingString = stateParts.slice(2, stateParts.length - 1).join(':');
    if (bettingString) {
      result.playerActionStrings = _getActionStringsFromACPC(
        bettingString,
        nextPlayerIndex,
        numPlayers
      );
    }
  }

  const odds = universalPokerJSON.odds || [];
  const p0WinOdds = Number(odds[0] ?? 0).toLocaleString(undefined, {
    style: 'percent',
    minimumFractionDigits: 2
  });
  const p1WinOdds = Number(odds[1] ?? 0).toLocaleString(undefined, {
    style: 'percent',
    minimumFractionDigits: 2
  });
  result.winOdds = [p0WinOdds, p1WinOdds];

  return result;
}

function splitCards(cardString) {
  if (!cardString) {
    return [];
  }
  return cardString.match(/.{1,2}/g) || [];
}

function isPlaceholderString(cardString) {
  return typeof cardString === 'string' && /^((2c)+)$/i.test(cardString);
}

function sanitizeCardList(cards) {
  return (cards || []).filter((card) => card);
}

function formatActionDisplay(actionString) {
  if (!actionString) {
    return '';
  }
  const playerMatch = actionString.match(/move=([^\s]+)/);
  if (!playerMatch) {
    return '';
  }
  const moveRaw = playerMatch[1];
  const moveLower = moveRaw.toLowerCase();
  if (moveLower.startsWith('bet') || moveLower.startsWith('raise')) {
    const amountMatch = moveRaw.match(/\d+/);
    return amountMatch ? `r${amountMatch[0]}` : 'r';
  }
  if (moveLower === 'call') {
    return 'c';
  }
  if (moveLower === 'check') {
    return 'k';
  }
  if (moveLower === 'fold') {
    return 'f';
  }
  return moveRaw;
}

function getBlinds(configuration) {
  const blindConfig = configuration?.openSpielGameParameters?.universal_poker_game_string?.blind;
  if (typeof blindConfig !== 'string') {
    return { bigBlind: null, smallBlind: null };
  }
  const parts = blindConfig
    .trim()
    .split(/\s+/)
    .map((entry) => Number(entry))
    .filter((n) => !Number.isNaN(n));
  if (parts.length >= 2) {
    return { bigBlind: parts[0], smallBlind: parts[1] };
  }
  return { bigBlind: null, smallBlind: null };
}

function getCommunityCardsFromUniversal(universal, numPlayers) {
  const parsed = _parseStepHistoryData(universal, null, numPlayers);
  const cards = splitCards(parsed.communityCards);
  const actual = cards.filter((card) => card && card.toLowerCase() !== PLACEHOLDER_CARD);
  if (actual.length < 3) {
    return [];
  }
  return actual;
}

function getHandCardsFromUniversal(universal, numPlayers) {
  const parsed = _parseStepHistoryData(universal, null, numPlayers);
  return (parsed.cards || []).map((cardString) => {
    if (isPlaceholderString(cardString)) {
      return [];
    }
    return sanitizeCardList(splitCards(cardString));
  });
}

function buildTimeline(environment, numPlayers) {
  const timeline = [];
  const stateHistory = environment?.info?.stateHistory || [];
  if (!stateHistory.length) {
    return timeline;
  }

  const parsedStates = stateHistory.map((entry, idx) => {
    const outer = JSON.parse(entry);
    return {
      idx,
      outer,
      universal: JSON.parse(outer.current_universal_poker_json)
    };
  });

  const { bigBlind, smallBlind } = getBlinds(environment.configuration);
  const dealer = parsedStates[0]?.outer?.dealer ?? 0;
  const smallBlindPlayer = dealer % numPlayers;
  const bigBlindPlayer = (dealer + 1) % numPlayers;

  const rawSteps = environment.__rawSteps || [];
  const flattenedActions = [];
  rawSteps.forEach((stepArr) => {
    (stepArr || []).forEach((agentStep, index) => {
      if (agentStep?.action && agentStep.action.submission !== -1) {
        flattenedActions.push({
          playerIndex: index,
          actionString: agentStep.action.actionString || ''
        });
      }
    });
  });

  const events = [];

  const addEvent = (event) => {
    events.push(event);
  };

  const initialStateIndex = parsedStates[0].idx;
  addEvent({ order: -30, stateIndex: initialStateIndex, highlightPlayer: null, actionText: '', hideHoleCards: true, hideCommunity: true });
  addEvent({ order: -20, stateIndex: initialStateIndex, highlightPlayer: smallBlindPlayer, actionText: smallBlind != null ? `SB ${smallBlind}` : 'SB', hideHoleCards: true, hideCommunity: true });
  addEvent({ order: -10, stateIndex: initialStateIndex, highlightPlayer: bigBlindPlayer, actionText: bigBlind != null ? `BB ${bigBlind}` : 'BB', hideHoleCards: true, hideCommunity: true });

  const firstActionIndex = parsedStates.findIndex(({ universal }) => universal.current_player !== -1);
  const holeEventState = firstActionIndex >= 0 ? parsedStates[firstActionIndex] : parsedStates[0];
  addEvent({ order: holeEventState.idx - 0.5, stateIndex: holeEventState.idx, highlightPlayer: null, actionText: '', hideHoleCards: false, hideCommunity: true });

  let statePointer = 0;
  flattenedActions.forEach((action) => {
    while (statePointer < parsedStates.length && parsedStates[statePointer].universal.current_player === -1) {
      statePointer++;
    }
    if (statePointer >= parsedStates.length) {
      return;
    }
    const preState = parsedStates[statePointer];
    const postState = parsedStates[statePointer + 1] ? parsedStates[statePointer + 1] : preState;
    const actionText = formatActionDisplay(action.actionString);
    addEvent({
      order: postState.idx,
      stateIndex: postState.idx,
      highlightPlayer: action.playerIndex,
      actionText,
      hideHoleCards: false,
      hideCommunity: false
    });
    statePointer++;
  });

  let currentStageCommunityLength = 0;
  parsedStates.forEach((stateInfo) => {
    const communityCards = getCommunityCardsFromUniversal(stateInfo.universal, numPlayers);
    const communityLength = communityCards.length;
    if (communityLength > currentStageCommunityLength) {
      const shouldDisplay = communityLength === 3 || communityLength === 4 || communityLength === 5;
      currentStageCommunityLength = communityLength;
      if (shouldDisplay) {
        addEvent({
          order: stateInfo.idx + 0.1,
          stateIndex: stateInfo.idx,
          highlightPlayer: null,
          actionText: '',
          hideHoleCards: false,
          hideCommunity: false,
          stage: true
        });
      }
    }
  });

  events.sort((a, b) => a.order - b.order);
  return events.map(({ stateIndex, highlightPlayer, actionText, hideHoleCards, hideCommunity }) => ({
    stateIndex,
    highlightPlayer,
    actionText,
    hideHoleCards: !!hideHoleCards,
    hideCommunity: !!hideCommunity
  }));
}

function getTimeline(environment, numPlayers) {
  if (!environment.__timeline) {
    environment.__timeline = buildTimeline(environment, numPlayers);
  }
  return environment.__timeline;
}

function getUniversalState(environment, index) {
  const entry = environment?.info?.stateHistory?.[index];
  if (!entry) {
    return null;
  }
  const outer = JSON.parse(entry);
  return {
    outer,
    universal: JSON.parse(outer.current_universal_poker_json)
  };
}

export const getPokerStateForStep = (environment, step) => {
  const numPlayers = 2;
  if (!environment || !environment.info?.stateHistory) {
    return null;
  }

  const timeline = getTimeline(environment, numPlayers);
  const event = timeline[step];
  if (!event) {
    return null;
  }

  const stateInfo = getUniversalState(environment, event.stateIndex);
  if (!stateInfo) {
    return null;
  }

  const parsedStateHistory = _parseStepHistoryData(
    stateInfo.universal,
    stateInfo.universal?.current_player,
    numPlayers
  );

  const startingStacks = stateInfo.universal?.starting_stacks || Array(numPlayers).fill(0);
  const contributions = stateInfo.universal?.player_contributions || parsedStateHistory.bets || Array(numPlayers).fill(0);
  const rewards = stateInfo.outer?.hand_returns || [];
  const communityCards = getCommunityCardsFromUniversal(stateInfo.universal, numPlayers);

  const players = Array(numPlayers)
    .fill(null)
    .map((_, i) => {
      const agentName = environment?.info?.TeamNames?.[i] || `Player ${i}`;
      const thumbnail = environment?.info?.Agents?.[i]?.ThumbnailUrl;
      return {
        id: `player${i}`,
        name: agentName,
        thumbnail,
        stack: startingStacks[i] - (contributions[i] || 0),
        cards: [],
        currentBet: contributions[i] || 0,
        isDealer: stateInfo.outer?.dealer === i,
        isTurn: stateInfo.universal?.current_player === i,
        isLastActor: event.highlightPlayer === i,
        reward: rewards[0]?.[i] ?? null,
        actionDisplayText: event.highlightPlayer === i ? event.actionText : ''
      };
    });

  const handCards = getHandCardsFromUniversal(stateInfo.universal, numPlayers);
  players.forEach((player, index) => {
    if (event.hideHoleCards) {
      player.cards = [];
    } else {
      player.cards = handCards[index] || [];
    }
  });

  const displayCommunity = event.hideCommunity ? [] : communityCards;

  return {
    players,
    communityCards: displayCommunity,
    pot: contributions.reduce((sum, value) => sum + (value || 0), 0),
    isTerminal: false,
    rawObservation: stateInfo.universal,
    step,
    winOdds: parsedStateHistory.winOdds,
    fiveCardBestHands: [],
    currentPlayer: stateInfo.universal?.current_player ?? -1,
    winner: -1
  };
};
