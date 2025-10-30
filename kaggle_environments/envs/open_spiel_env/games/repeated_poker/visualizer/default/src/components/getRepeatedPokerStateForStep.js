import { getActionStringsFromACPC } from '@kaggle-environments/core';

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

  const event = environment.steps[step];

  if (!event) {
    return null;
  }

  const stateInfo = getUniversalState(environment, event.stateHistoryIndex);
  if (!stateInfo) {
    return null;
  }

  const parsedStateHistory = _parseStepHistoryData(
    stateInfo.universal,
    event.activePlayer,
    numPlayers
  );

  const startingStacks = stateInfo.universal?.starting_stacks || Array(numPlayers).fill(0);
  const contributions =
    stateInfo.universal?.player_contributions || parsedStateHistory.bets || Array(numPlayers).fill(0);
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
        isTurn: event.activePlayer === i,
        isLastActor: event.highlightPlayer === i,
        reward: rewards[0]?.[i] ?? null,
        actionDisplayText: parsedStateHistory.playerActionStrings[i],
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
