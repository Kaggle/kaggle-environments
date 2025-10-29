import { PokerGameStep } from "../types";

const _isStateHistoryAgentAction = (stateHistoryEntry: string): boolean =>
  JSON.parse(JSON.parse(stateHistoryEntry).current_universal_poker_json)
    .current_player !== -1;
const _isStateHistoryEntryInitial = (stateHistoryEntry: string): boolean => {
  const state = JSON.parse(
    JSON.parse(stateHistoryEntry).current_universal_poker_json,
  );
  return state.acpc_state.startsWith("STATE:0::2c2c|2c2c");
};
export const getMoveHistoryFromACPC = (acpcState: string): string => {
  // Parse the ACPC state line to extract the betting string
  // Example ACPC state: "STATE:0:r5c/cr11c/:6cKd|AsJc/7hQh6d/2c"
  const lines = acpcState.trim().split("\n");
  if (lines.length < 1) {
    return "";
  }
  const stateLine = lines[0]; // First line contains the state
  const stateParts = stateLine.split(":");
  // The betting string is everything between the 2nd colon and the last colon
  // stateParts[0] = "STATE"
  // stateParts[1] = "0" (hand number)
  // stateParts[2...-1] = betting string
  // stateParts[last] = cards
  if (stateParts.length < 4) {
    return "";
  }
  const bettingString = stateParts.slice(2, stateParts.length - 1).join(":");
  return bettingString;
};

function _getMovesFromBettingStringACPC(bettingString: string): string[] {
  const moves = [];
  // Split the action string by street (e.g., ["r5c", "cr11f"])
  const streets = bettingString.split("/");
  // Process each street's actions
  for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
    const streetAction = streets[streetIndex];
    let i = 0;
    while (i < streetAction.length) {
      const char = streetAction[i];
      if (char === "r") {
        // 'r' (raise)
        let amount = "";
        i++;
        // Continue to parse all digits of the raise amount
        while (
          i < streetAction.length &&
          streetAction[i] >= "0" &&
          streetAction[i] <= "9"
        ) {
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

export function _getReadableMovesFromBettingStringACPC(
  bettingString: string,
): string[] {
  if (!bettingString) {
    return [];
  }

  const moves: string[] = [];
  const streets = bettingString.split("/");
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

    let actingPlayer =
      FIRST_ACTOR_BY_STREET[Math.min(streetIndex, FIRST_ACTOR_BY_STREET.length - 1)];

    let i = 0;
    while (i < trimmedAction.length) {
      const char = trimmedAction[i];
      const currentMax = Math.max(...totalContributions);

      if (char === "r") {
        let amount = "";
        i++;
        while (
          i < trimmedAction.length &&
          trimmedAction[i] >= "0" &&
          trimmedAction[i] <= "9"
        ) {
          amount += trimmedAction[i];
          i++;
        }
        const targetTotal = parseInt(amount || "0", 10);
        const previousTotal = totalContributions[actingPlayer];
        const roundBaseline = streetBaselines[actingPlayer];
        const roundTotal = Math.max(targetTotal - roundBaseline, 0);
        const hasOutstandingBet = currentMax > previousTotal;
        const verb = hasOutstandingBet ? "Raise" : "Bet";

        if (!Number.isFinite(targetTotal)) {
          throw new Error(
            `Invalid raise amount '${amount}' parsed from betting string '${bettingString}'.`,
          );
        }
        if (targetTotal <= previousTotal) {
          throw new Error(
            `Invalid raise target ${targetTotal} for player ${actingPlayer} (previous total ${previousTotal}).`,
          );
        }

        moves.push(`${verb} ${roundTotal}`);
        totalContributions[actingPlayer] = targetTotal;
      } else if (char === "c") {
        const previousTotal = totalContributions[actingPlayer];
        if (previousTotal === currentMax) {
          moves.push("Check");
        } else {
          const callAmount = currentMax - previousTotal;
          moves.push(callAmount > 0 ? `Call ${callAmount}` : "Call");
          totalContributions[actingPlayer] = currentMax;
        }
        i++;
      } else if (char === "f") {
        moves.push("Fold");
        i++;
      } else {
        throw new Error(
          `Unknown betting token '${char}' encountered in '${bettingString}'.`,
        );
      }

      actingPlayer = (actingPlayer + 1) % NUM_PLAYERS;
    }
  });

  return moves;
}

const _getEndCondition = (
  stateHistory: any[],
  stateHistoryPointer: number,
  currentPlayer: string,
): {
  handConclusion: "fold" | "showdown";
  winner: -1 | 0 | 1; // -1 for the rare event of a tie
  bestFiveCardHands?: string[];
  bestHandRankType?: string[];
} => {
  const current_player = parseInt(currentPlayer);
  if (stateHistoryPointer >= stateHistory.length - 1) {
    return {
      // TODO: handle tail end
      // for now, fold + tie = impossible state
      handConclusion: "fold",
      winner: -1,
      bestFiveCardHands: [],
    };
  }
  let next_prev_universal_poker_json = {
    acpc_state: "",
    best_five_card_hands: ["", ""],
    best_hand_rank_types: ["", ""],
  };
  // since the current_universal_poker_json does not contain the end move in it's history,
  // we need to go to the prev_universal_poker_json of the next one
  try {
    next_prev_universal_poker_json = JSON.parse(
      JSON.parse(stateHistory[stateHistoryPointer + 1])
        .prev_universal_poker_json,
    );
  } catch {
    console.error("prev_universal_poker_json parsing failed");
  }
  // if the stateHistory doesn't end in a fold, it was a showdown
  const bettingString = getMoveHistoryFromACPC(
    next_prev_universal_poker_json.acpc_state,
  );
  const moves = _getMovesFromBettingStringACPC(bettingString);
  // Fold case
  if (moves.pop() === "f") {
    return {
      handConclusion: "fold",
      winner: current_player === 0 ? 1 : 0,
    };
  }
  // Showdown case
  return {
    handConclusion: "showdown",
    winner: current_player === 0 ? 1 : 0,
    bestFiveCardHands: next_prev_universal_poker_json.best_five_card_hands,
    bestHandRankType: next_prev_universal_poker_json.best_hand_rank_types,
  };
};

export const getPokerStepLabel = (gameStep: PokerGameStep) => {
  const decision = gameStep.step?.action?.actionString ?? "";
  if (decision.length > 0) {
    return decision
      .split("move=")[1]
      .replace(/([a-zA-Z])(\d)/g, "$1 $2")
      .replace(/(\d)([A-Z])/g, "$1 $2");
  }

  return "";
};

export const getPokerStepDescription = (gameStep: PokerGameStep) => {
  if (gameStep.step?.action?.thoughts) {
    return gameStep.step.action.thoughts;
  } else if (gameStep.isEndState) {
    if (gameStep.handConclusion === "showdown") {
      return `
## Player ${gameStep.winner} wins round ${gameStep.hand + 1} 
### Wins with ${gameStep.bestHandRankType} 
### ${gameStep.bestFiveCardHands} 
`;
    } else {
      return `
## Player ${gameStep.winner} wins round ${gameStep.hand + 1} 
### Other player folds
`;
    }
  }

  // TODO player names
  return "TODO";
};

/* interface TimelineEvent {
  stateIndex: number | undefined;
  highlightPlayer: number | null;
  actionText: string;
  hideHoleCards: boolean;
  hideCommunity: boolean;
} */

export const getPokerStepsWithEndStates = (
  environment: any,
): PokerGameStep[] => {
  const stepsWithEndStates: PokerGameStep[] = [];
  let handCount = 0;
  let stateHistoryPointer = 0;

  const stateHistory = environment.info.stateHistory ?? [];
  const steps = environment.steps ?? [];

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

          stepsWithEndStates.push({
            hand: handCount,
            isEndState: false,
            step: s,
            stateHistory: stateHistory[stateHistoryPointer],
            stateHistoryIndex: preActionPointer,
          });

          lastActionPointer = preActionPointer;
          stateHistoryPointer++;

          const postActionPointer = stateHistoryPointer;

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
      (lookaheadPointer >= stateHistory.length ||
        _isStateHistoryEntryInitial(stateHistory[lookaheadPointer]));

    if (isEndState) {
      const endState = _getEndCondition(
        stateHistory,
        lastActionPointer,
        step[0].observation.currentPlayer,
      );

      stepsWithEndStates.push({
        hand: handCount,
        isEndState: true,
        step: null,
        stateHistory: stateHistory[lastActionPointer],
        stateHistoryIndex: lastActionPointer,
        ...endState,
      });

      handCount++;
    }
  }

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
  */

  return stepsWithEndStates;
};

export const __testing = {
  _getMovesFromBettingStringACPC,
  _getReadableMovesFromBettingStringACPC,
};
