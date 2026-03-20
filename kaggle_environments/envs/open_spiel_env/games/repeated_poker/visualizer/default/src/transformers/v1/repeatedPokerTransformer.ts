import { RepeatedPokerStep, RepeatedPokerStepPlayer } from '../v2/poker-steps-types';
import { getActionStringsFromACPC } from './buildTimeline';

// const _parseRoundState = (currentStateHistory: string) => {
//   const currentState = JSON.parse(JSON.parse(currentStateHistory).current_universal_poker_json).acpc_state;
//
//   /**
//    * Example lines:
//    * STATE:0:r5c/cr9c/:Ks4s|5hAs/2dJs7s/Qh
//    * Spent: [P0: 9  P1: 9  ]
//    */
//   const lines = currentState.trim().split('\n');
//   if (lines.length < 2) {
//     return '';
//   }
//   const stateParts = lines[0].split(':');
//
//   const currentCardString = stateParts[stateParts.length - 1]; // example: "6cKd|AsJc/7hQh6d/2c"
//   // Grab the hand and board blocks
//   const currentCardSegments = currentCardString.split('|');
//   // Split card string by '/' to separate hand and board blocks
//   const currentCommunitySegments = currentCardSegments.length > 1 ? currentCardSegments[1].split('/') : [];
//
//   if (currentCommunitySegments.length === 2) {
//     return '### Flop';
//   } else if (currentCommunitySegments.length === 3) {
//     return '### 4th Street';
//   } else if (currentCommunitySegments.length === 4) {
//     return '### 5th Street';
//   } else {
//     return '';
//   }
// };

const _isStateHistoryAgentAction = (stateHistoryEntry: string): boolean =>
  JSON.parse(JSON.parse(stateHistoryEntry).current_universal_poker_json).current_player !== -1;
const _isStateHistoryEntryInitial = (stateHistoryEntry: string): boolean => {
  const state = JSON.parse(JSON.parse(stateHistoryEntry).current_universal_poker_json);
  return state.acpc_state.startsWith('STATE:0::2c2c|2c2c');
};
export const getMoveHistoryFromACPC = (acpcState: string): string => {
  // Parse the ACPC state line to extract the betting string
  // Example ACPC state: "STATE:0:r5c/cr11c/:6cKd|AsJc/7hQh6d/2c"
  const lines = acpcState.trim().split('\n');
  if (lines.length < 1) {
    return '';
  }
  const stateLine = lines[0]; // First line contains the state
  const stateParts = stateLine.split(':');
  // The betting string is everything between the 2nd colon and the last colon
  // stateParts[0] = "STATE"
  // stateParts[1] = "0" (hand number)
  // stateParts[2...-1] = betting string
  // stateParts[last] = cards
  if (stateParts.length < 4) {
    return '';
  }
  const bettingString = stateParts.slice(2, stateParts.length - 1).join(':');
  return bettingString;
};

function _getMovesFromBettingStringACPC(bettingString: string): string[] {
  const moves = [];
  // Split the action string by street (e.g., ["r5c", "cr11f"])
  const streets = bettingString.split('/');
  // Process each street's actions
  for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
    const streetAction = streets[streetIndex];
    let i = 0;
    while (i < streetAction.length) {
      const char = streetAction[i];
      if (char === 'r') {
        // 'r' (raise)
        let amount = '';
        i++;
        // Continue to parse all digits of the raise amount
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
  return moves;
}

export function _getReadableMovesFromBettingStringACPC(bettingString: string): string[] {
  if (!bettingString) {
    return [];
  }

  const moves: string[] = [];
  const streets = bettingString.split('/');
  // Heads-up specific ordering: SB acts first preflop, BB acts first postflop
  const FIRST_ACTOR_BY_STREET = [1, 0, 0, 0];
  const NUM_PLAYERS = 2;

  // Track the total amount contributed by each player across the hand.
  // Start with blinds posted (BB=2, SB=1) for repeated poker.
  const totalContributions: number[] = [2, 1];
  // Track the contributions each player had at the start of the street.
  // Preflop baseline excludes blinds so that we report the amount invested during the action.
  let streetBaselines: number[] = [0, 0];

  streets.forEach((streetAction, streetIndex) => {
    const trimmedAction = streetAction.trim();
    // Update baselines for every street after preflop
    if (streetIndex > 0) {
      streetBaselines = [...totalContributions];
    }

    if (!trimmedAction) {
      return;
    }

    let actingPlayer = FIRST_ACTOR_BY_STREET[Math.min(streetIndex, FIRST_ACTOR_BY_STREET.length - 1)];

    let i = 0;
    while (i < trimmedAction.length) {
      const char = trimmedAction[i];
      const currentMax = Math.max(...totalContributions);
      const highestBaseline = Math.max(...streetBaselines);

      if (char === 'r') {
        let amount = '';
        i++;
        while (i < trimmedAction.length && trimmedAction[i] >= '0' && trimmedAction[i] <= '9') {
          amount += trimmedAction[i];
          i++;
        }
        const targetTotal = parseInt(amount || '0', 10);
        const previousTotal = totalContributions[actingPlayer];
        const roundBaseline = streetBaselines[actingPlayer];
        const roundTotal = Math.max(targetTotal - roundBaseline, 0);
        const hasOutstandingBet = currentMax > highestBaseline;
        const verb = hasOutstandingBet ? 'Raise' : 'Bet';

        if (!Number.isFinite(targetTotal)) {
          throw new Error(`Invalid raise amount '${amount}' parsed from betting string '${bettingString}'.`);
        }
        if (targetTotal <= previousTotal) {
          throw new Error(
            `Invalid raise target ${targetTotal} for player ${actingPlayer} (previous total ${previousTotal}).`
          );
        }

        moves.push(`${verb} ${roundTotal}`);
        totalContributions[actingPlayer] = targetTotal;
      } else if (char === 'c') {
        const previousTotal = totalContributions[actingPlayer];
        if (previousTotal === currentMax) {
          moves.push('Check');
        } else {
          const callAmount = currentMax - previousTotal;
          moves.push(callAmount > 0 ? `Call ${callAmount}` : 'Call');
          totalContributions[actingPlayer] = currentMax;
        }
        i++;
      } else if (char === 'f') {
        moves.push('Fold');
        i++;
      } else {
        throw new Error(`Unknown betting token '${char}' encountered in '${bettingString}'.`);
      }

      actingPlayer = (actingPlayer + 1) % NUM_PLAYERS;
    }
  });

  return moves;
}

const _getEndCondition = (
  stateHistory: any[],
  stateHistoryPointer: number,
  currentPlayer: string
): {
  handConclusion: 'fold' | 'showdown';
  winner: -1 | 0 | 1; // -1 for the rare event of a tie
  bestFiveCardHands?: string[];
  bestHandRankType?: string[];
} => {
  const current_player = parseInt(currentPlayer);
  if (stateHistoryPointer >= stateHistory.length - 1) {
    return {
      // TODO: handle tail end
      // for now, fold + tie = impossible state
      handConclusion: 'fold',
      winner: -1,
      bestFiveCardHands: [],
    };
  }
  let next_prev_universal_poker_json = {
    acpc_state: '',
    best_five_card_hands: ['', ''],
    best_hand_rank_types: ['', ''],
  };
  // since the current_universal_poker_json does not contain the end move in it's history,
  // we need to go to the prev_universal_poker_json of the next one
  try {
    next_prev_universal_poker_json = JSON.parse(
      JSON.parse(stateHistory[stateHistoryPointer + 1]).prev_universal_poker_json
    );
  } catch {
    console.error('prev_universal_poker_json parsing failed');
  }
  // if the stateHistory doesn't end in a fold, it was a showdown
  const bettingString = getMoveHistoryFromACPC(next_prev_universal_poker_json.acpc_state);
  const moves = _getMovesFromBettingStringACPC(bettingString);
  // Fold case
  if (moves.pop() === 'f') {
    return {
      handConclusion: 'fold',
      winner: current_player === 0 ? 1 : 0,
    };
  }
  // Showdown case
  return {
    handConclusion: 'showdown',
    winner: current_player === 0 ? 1 : 0,
    bestFiveCardHands: next_prev_universal_poker_json.best_five_card_hands,
    bestHandRankType: next_prev_universal_poker_json.best_hand_rank_types,
  };
};

export const getPokerStepLabel = (gameStep: RepeatedPokerStep) => {
  const currentHandNumber = gameStep.currentHandIndex + 1;
  switch (gameStep.stepType) {
    case 'player-action':
      return `**Decision**: ${gameStep.players[gameStep.currentPlayer].actionDisplayText ?? ''}`;
    case 'deal-player-hands':
      return `**Hand ${currentHandNumber}**: Dealing...`;
    case 'deal-flop':
      return `**Hand ${currentHandNumber}**: Flop`;
    case 'deal-turn':
      return `**Hand ${currentHandNumber}**: Turn`;
    case 'deal-river':
      return `**Hand ${currentHandNumber}**: River`;
    case 'big-blind-post':
      return `**Hand ${currentHandNumber}**: Post Big Blind`;
    case 'small-blind-post':
      return `**Hand ${currentHandNumber}**: Post Small Blind`;
    case 'final': {
      const winners = (gameStep.players as RepeatedPokerStepPlayer[]).filter((p) => p.isWinner);
      return winners.length === 1
        ? `**Hand ${currentHandNumber}**: 🎉 ${winners[0].name} wins ${winners[0].reward}! 🎉`
        : `**Hand ${currentHandNumber}**: Split Pot`;
    }
    case 'game-over': {
      const winningPlayer = (gameStep.players as RepeatedPokerStepPlayer[]).find((p) => p.isWinner);
      if (winningPlayer) {
        return `🎉🎉🎉  ${winningPlayer?.name} wins the match! 🎉🎉🎉  `;
      } else {
        return 'MATCH IS A DRAW';
      }
    }
    default: {
      // If you missed a case, TypeScript will complain here because
      // it cannot assign the missed type to 'never'.
      const _exhaustiveCheck: never = gameStep.stepType;
      return `Unknown step: ${_exhaustiveCheck}`;
    }
  }
};

export const getPokerStepDescription = (gameStep: RepeatedPokerStep) => {
  switch (gameStep.stepType) {
    case 'player-action':
      return gameStep.players[gameStep.currentPlayer].thoughts ?? '';
    case 'deal-player-hands':
      return '';
    case 'deal-flop':
      return '';
    case 'deal-turn':
      return '';
    case 'deal-river':
      return '';
    case 'big-blind-post':
      return '';
    case 'small-blind-post':
      return '';
    case 'final':
      return '';
    case 'game-over':
      return '';
    default: {
      // If you missed a case, TypeScript will complain here because
      // it cannot assign the missed type to 'never'.
      const _exhaustiveCheck: never = gameStep.stepType;
      return `Unknown step: ${_exhaustiveCheck}`;
    }
  }
};

export const getPokerStepsWithEndStates = (environment: any): any[] => {
  const stepsWithEndStates: any[] = [];
  let handCount = 0;
  let stateHistoryPointer = 0;

  const stateHistory = environment.info.stateHistory ?? [];
  stateHistory.forEach((entry: string, idx: number) => {
    if (idx === 74) {
      console.log(JSON.parse(entry));
      console.log('!!!!');
      console.log(JSON.parse(JSON.parse(entry).current_universal_poker_json));
    }
  });
  const steps = environment.steps ?? [];
  const teamNames: string[] = Array.isArray(environment?.info?.TeamNames)
    ? environment.info.TeamNames
    : Array.isArray(environment?.info?.Names)
      ? environment.info.Names
      : [];

  const normalizePlayerIndex = (value: any): number | null => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === 'string' && value.trim().length > 0) {
      const parsed = parseInt(value, 10);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
    return null;
  };

  const getPlayerName = (index: number | null): string | null => {
    if (index === null || index === undefined || index < 0) {
      return null;
    }
    return teamNames[index] ?? `Player ${index}`;
  };

  const extractCurrentPlayerFromStateHistory = (index: number | null): number | null => {
    if (index === null || index === undefined) {
      return null;
    }
    if (index < 0 || index >= stateHistory.length) {
      return null;
    }
    try {
      const outer = JSON.parse(stateHistory[index]);
      const universal = JSON.parse(outer.current_universal_poker_json ?? 'null');
      const outerCurrent = normalizePlayerIndex(outer?.current_player);
      if (outerCurrent !== null) {
        return outerCurrent;
      }
      return normalizePlayerIndex(universal?.current_player ?? null);
    } catch {
      return null;
    }
  };

  const advanceToNextAgentEntry = () => {
    while (
      stateHistoryPointer < stateHistory.length &&
      !_isStateHistoryAgentAction(stateHistory[stateHistoryPointer])
    ) {
      stateHistoryPointer++;
    }
  };

  advanceToNextAgentEntry();

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    let lastActionPointer = -1;

    if (step) {
      step.forEach((s: any) => {
        if (s.action.submission !== -1) {
          if (stateHistoryPointer >= stateHistory.length) {
            return;
          }

          const preActionPointer = stateHistoryPointer;
          const actionString: string = s?.action?.actionString ?? '';
          const playerMatch = actionString.match(/player=(\d+)/);
          const actingPlayer = playerMatch
            ? normalizePlayerIndex(playerMatch[1])
            : normalizePlayerIndex(s?.observation?.playerId);
          const currentPlayer = normalizePlayerIndex(s?.observation?.currentPlayer);

          stepsWithEndStates.push({
            hand: handCount,
            isEndState: false,
            step: s,
            stateHistory: stateHistory[stateHistoryPointer],
            stateHistoryIndex: preActionPointer,
            actingPlayer,
            actingPlayerName: getPlayerName(actingPlayer),
            currentPlayer,
            currentPlayerName: getPlayerName(currentPlayer),
          });

          lastActionPointer = preActionPointer;
          stateHistoryPointer++;

          const postActionPointer = stateHistoryPointer;
          const postActionCurrentPlayer = extractCurrentPlayerFromStateHistory(postActionPointer);

          if (
            postActionPointer < stateHistory.length &&
            !_isStateHistoryAgentAction(stateHistory[postActionPointer]) &&
            !_isStateHistoryEntryInitial(stateHistory[postActionPointer])
          ) {
            stepsWithEndStates.push({
              hand: handCount,
              isEndState: false,
              step: null,
              stateHistory: stateHistory[postActionPointer],
              stateHistoryIndex: postActionPointer,
              postActionOf: preActionPointer,
              actingPlayer,
              actingPlayerName: getPlayerName(actingPlayer),
              currentPlayer: postActionCurrentPlayer,
              currentPlayerName: getPlayerName(postActionCurrentPlayer),
            });
          }

          advanceToNextAgentEntry();
        }
      });
    }

    let lookaheadPointer = lastActionPointer + 1;
    while (
      lookaheadPointer < stateHistory.length &&
      !_isStateHistoryAgentAction(stateHistory[lookaheadPointer]) &&
      !_isStateHistoryEntryInitial(stateHistory[lookaheadPointer])
    ) {
      lookaheadPointer++;
    }

    const isEndState =
      lastActionPointer !== -1 &&
      (lookaheadPointer >= stateHistory.length || _isStateHistoryEntryInitial(stateHistory[lookaheadPointer]));

    if (isEndState) {
      const endStateCurrentPlayer = extractCurrentPlayerFromStateHistory(lastActionPointer);
      const endState = _getEndCondition(stateHistory, lastActionPointer, endStateCurrentPlayer?.toString() ?? '');

      stepsWithEndStates.push({
        hand: handCount,
        isEndState: true,
        step: null,
        stateHistory: stateHistory[lastActionPointer],
        stateHistoryIndex: lastActionPointer,
        currentPlayer: endStateCurrentPlayer,
        currentPlayerName: getPlayerName(endStateCurrentPlayer),
        ...endState,
      });

      handCount++;
    }
  }

  // After building the original stepsWithEndStates, add action strings to each step
  const enhancedSteps = stepsWithEndStates.map((step) => {
    try {
      // Only process steps that have stateHistory
      if (!step.stateHistory) {
        return step;
      }

      const outer = JSON.parse(step.stateHistory);
      const universal = JSON.parse(outer.current_universal_poker_json ?? 'null');

      // Extract betting string from universal poker JSON
      const bettingString = getMoveHistoryFromACPC(universal?.acpc_state || '');

      // Get the next player index for this state
      const nextPlayerIndex = normalizePlayerIndex(universal?.current_player ?? null);

      // Get action strings for each player
      const playerActionStrings = getActionStringsFromACPC(
        bettingString,
        2 // Assuming 2 players for poker
      );

      // Return a new step with action strings added
      return {
        ...step,
        actionText: nextPlayerIndex ? playerActionStrings[nextPlayerIndex] : undefined,
      };
    } catch (error) {
      console.error('Error adding action strings to step:', error);
      // If there's an error, return the original step
      return step;
    }
  });

  return enhancedSteps;

  /*
  // Build timeline from the environment
  const timeline: TimelineEvent[] = buildTimeline(
    environment,
    2,
  );

  // Create new steps based on the timeline
  const timelineSteps = timeline.map(
    (timelineEvent: TimelineEvent): PokerGameStep => {
      // Find the corresponding step in stepsWithEndStates by matching stateHistoryIndex
      const matchingStep = stepsWithEndStates.find(
        (step) => step.stateHistoryIndex === timelineEvent.stateIndex,
      );

      if (!matchingStep) {
        // Find the hand number by looking at the closest previous step
        let handNumber = 0;
        if (timelineEvent.stateIndex !== undefined) {
          const previousSteps = stepsWithEndStates.filter(
            (step) =>
              step.stateHistoryIndex !== undefined &&
              timelineEvent.stateIndex !== undefined &&
              step.stateHistoryIndex < timelineEvent.stateIndex,
          );
          if (previousSteps.length > 0) {
            handNumber = previousSteps[previousSteps.length - 1].hand;
          }
        }

        // Create a new step if there's no matching step
        return {
          hand: handNumber,
          isEndState: false,
          step: null,
          stateHistory:
            timelineEvent.stateIndex !== undefined
              ? stateHistory[timelineEvent.stateIndex]
              : "",
          stateHistoryIndex: timelineEvent.stateIndex,
          actionText: timelineEvent.actionText,
        };
      }

      // Return a new step with timeline data included
      return {
        ...matchingStep,
        actionText: timelineEvent.actionText,
      };
    },
  );

  return stepsWithEndStates;
  */
};

export const __testing = {
  _getMovesFromBettingStringACPC,
  _getReadableMovesFromBettingStringACPC,
};
