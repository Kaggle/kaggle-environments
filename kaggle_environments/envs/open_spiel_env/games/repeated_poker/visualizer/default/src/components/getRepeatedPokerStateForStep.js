import { getActionStringsFromACPC } from '../../../../../../../../../web/core/dist/transformers/buildTimeline.js';
import { getPokerStepsWithEndStates } from '../../../../../../../../../web/core/dist/transformers/repeatedPokerTransformer.js';

const PLACEHOLDER_CARD = '2c';

function _parseStepHistoryData(universalPokerJSON, nextPlayerIndex, numPlayers = 2) {
  const result = {
    cards: [],
    communityCards: '',
    bets: [],
    playerActionStrings: Array(numPlayers).fill(''),
    winOdds: [0, 0]
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

    result.communityCards = cardSegments.slice(1).filter(Boolean).join('');

    const bettingString = stateParts.slice(2, stateParts.length - 1).join(':');
    if (bettingString) {
      result.playerActionStrings = getActionStringsFromACPC(bettingString, nextPlayerIndex, numPlayers);
    }
  }

  const odds = universalPokerJSON.odds || [];
  // The odds array is structured as [Player1_Win_Prob, Tie_Prob, Player2_Win_Prob, Tie_Prob_Repeated]
  const p0WinProb = Number(odds[0] ?? 0);
  const tieProb = Number(odds[1] ?? 0);
  const p1WinProb = Number(odds[2] ?? 0);
  const fiveCardBestHands = universalPokerJSON.best_hand_rank_types || [];

  result.winProb = [
    p0WinProb.toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 }),
    p1WinProb.toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 })
  ];
  result.tieProb = tieProb.toLocaleString(undefined, {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1
  });
  result.handRank = fiveCardBestHands;

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

function extractStateHistory(environment) {
  return (
    environment?.info?.stateHistory ||
    environment?.info?.state_history ||
    environment?.stateHistory ||
    environment?.state_history ||
    []
  );
}

function ensurePokerEnvironment(environment) {
  if (!environment || !Array.isArray(environment.steps)) {
    return { environment, stepMap: null };
  }

  if (!Array.isArray(environment.steps[0])) {
    // Already transformed into PokerGameStep objects.
    return { environment, stepMap: environment.__pokerStepMap || null };
  }

  if (environment.__pokerEnvironmentCache) {
    return environment.__pokerEnvironmentCache;
  }

  const stateHistory = extractStateHistory(environment);
  if (!Array.isArray(stateHistory) || stateHistory.length === 0) {
    const cacheEntry = { environment, stepMap: null };
    environment.__pokerEnvironmentCache = cacheEntry;
    return cacheEntry;
  }

  const derivedEnvironment = {
    configuration: environment.configuration ?? null,
    info: {
      ...(environment.info || {}),
      stateHistory
    },
    steps: []
  };

  const stepsWithEndStates = getPokerStepsWithEndStates({
    configuration: derivedEnvironment.configuration,
    info: derivedEnvironment.info,
    steps: environment.steps || []
  });

  derivedEnvironment.steps = stepsWithEndStates;

  const rawStepRefs = new Map();
  (environment.steps || []).forEach((rawStep, rawIndex) => {
    if (Array.isArray(rawStep)) {
      rawStep.forEach((entry) => {
        if (entry && typeof entry === 'object') {
          rawStepRefs.set(entry, rawIndex);
        }
      });
    }
  });

  const stepMap = new Map();
  stepsWithEndStates.forEach((entry, index) => {
    if (entry?.step && rawStepRefs.has(entry.step)) {
      const rawIndex = rawStepRefs.get(entry.step);
      if (!stepMap.has(rawIndex)) {
        stepMap.set(rawIndex, index);
      }
    }
  });

  let lastKnownIndex = 0;
  for (let i = 0; i < (environment.steps?.length || 0); i += 1) {
    if (stepMap.has(i)) {
      lastKnownIndex = stepMap.get(i);
    } else {
      stepMap.set(i, lastKnownIndex);
    }
  }

  const cacheEntry = { environment: derivedEnvironment, stepMap };
  environment.__pokerEnvironmentCache = cacheEntry;
  return cacheEntry;
}

function getCurrentStreetContributions(universalPokerJSON, numPlayers) {
  if (!universalPokerJSON) {
    return Array(numPlayers).fill(0);
  }

  const blinds = Array.from({ length: numPlayers }, (_, idx) => {
    const blindValue = universalPokerJSON.blinds?.[idx];
    const numericBlind = Number(blindValue);
    return Number.isFinite(numericBlind) ? numericBlind : 0;
  });

  const contributions = blinds.slice();
  let streetBaseline = Array(numPlayers).fill(0);
  const bettingHistory = universalPokerJSON.betting_history || '';
  const streets = bettingHistory.length > 0 ? bettingHistory.split('/') : [''];
  const FIRST_ACTOR_BY_STREET = [1, 0, 0, 0];

  streets.forEach((streetAction, streetIndex) => {
    if (streetIndex > 0) {
      streetBaseline = contributions.slice();
    }

    const trimmedAction = streetAction.trim();
    if (!trimmedAction) {
      return;
    }

    let actingPlayer =
      FIRST_ACTOR_BY_STREET[Math.min(streetIndex, FIRST_ACTOR_BY_STREET.length - 1)];

    let i = 0;
    while (i < trimmedAction.length) {
      const char = trimmedAction[i];
      const currentMax = Math.max(...contributions);

      if (char === 'r') {
        let amount = '';
        i += 1;
        while (i < trimmedAction.length && trimmedAction[i] >= '0' && trimmedAction[i] <= '9') {
          amount += trimmedAction[i];
          i += 1;
        }
        const targetTotal = parseInt(amount || '0', 10);
        if (Number.isFinite(targetTotal)) {
          contributions[actingPlayer] = targetTotal;
        }
      } else if (char === 'c') {
        if (contributions[actingPlayer] < currentMax) {
          contributions[actingPlayer] = currentMax;
        }
        i += 1;
      } else if (char === 'f') {
        i += 1;
      } else {
        i += 1;
      }

      actingPlayer = (actingPlayer + 1) % numPlayers;
    }
  });

  const finalTotals = Array.isArray(universalPokerJSON.player_contributions)
    ? universalPokerJSON.player_contributions.slice(0, numPlayers)
    : contributions;

  return finalTotals.map((value, idx) => Math.max(value - (streetBaseline[idx] || 0), 0));
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

  const { environment: derivedEnvironment, stepMap } = ensurePokerEnvironment(environment);
  if (!derivedEnvironment || !Array.isArray(derivedEnvironment.steps)) {
    return null;
  }

  const mappedStep =
    (stepMap && stepMap.has(step) ? stepMap.get(step) : null) ??
    Math.min(step, Math.max(derivedEnvironment.steps.length - 1, 0));

  const event = derivedEnvironment.steps[mappedStep];

  if (!event) {
    return null;
  }

  const stateInfo = getUniversalState(derivedEnvironment, event.stateHistoryIndex);
  if (!stateInfo) {
    return null;
  }

  const parsedStateHistory = _parseStepHistoryData(
    stateInfo.universal,
    stateInfo.universal?.current_player,
    numPlayers
  );

  const startingStacks = stateInfo.universal?.starting_stacks || Array(numPlayers).fill(0);
  const contributions =
    stateInfo.universal?.player_contributions || parsedStateHistory.bets || Array(numPlayers).fill(0);
  const currentStreetBets = getCurrentStreetContributions(stateInfo.universal, numPlayers);
  const rewards = stateInfo.outer?.hand_returns || [];
  const communityCards = getCommunityCardsFromUniversal(stateInfo.universal, numPlayers);
  const actionStrings = Array.isArray(parsedStateHistory.playerActionStrings)
    ? parsedStateHistory.playerActionStrings
    : Array(numPlayers).fill('');

  const players = Array(numPlayers)
    .fill(null)
    .map((_, i) => {
      const agentName = environment?.info?.TeamNames?.[i] || `Player ${i}`;
      const thumbnail = environment?.info?.Agents?.[i]?.ThumbnailUrl;
      const actionText = typeof actionStrings[i] === 'string' ? actionStrings[i] : '';
      const actedThisStep =
        (event?.actingPlayer !== undefined && event.actingPlayer === i) ||
        (event?.highlightPlayer !== undefined && event.highlightPlayer === i) ||
        actionText.length > 0;
      return {
        id: `player${i}`,
        name: agentName,
        thumbnail,
        stack: startingStacks[i] - (contributions[i] || 0),
        cards: [],
        currentBet: currentStreetBets[i] || 0,
        isDealer: stateInfo.outer?.dealer === i,
        isTurn: stateInfo.universal?.current_player === i,
        isLastActor: actedThisStep,
        reward: rewards[0]?.[i] ?? null,
        actionDisplayText: actionText,
        handCount: 0
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
    winProb: parsedStateHistory.winProb,
    tieProb: parsedStateHistory.tieProb,
    handRank: parsedStateHistory.handRank,
    currentPlayer: stateInfo.universal?.current_player ?? -1,
    winner: -1,
    handCount: event.hand,
  };
};
