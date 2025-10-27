
const _isStateHistoryAgentAction = (stateHistoryEntry: string): boolean =>
  JSON.parse(JSON.parse(stateHistoryEntry).current_universal_poker_json).current_player !== -1;

const _isStateHistoryEntryInitial = (stateHistoryEntry: string): boolean => {
  const state = JSON.parse(JSON.parse(stateHistoryEntry).current_universal_poker_json);
  return state.acpc_state.startsWith('STATE:0::2c2c|2c2c');
}

const _getMoveHistoryFromACPC = (acpcState: string): string => {
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
}

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
        moves.push(char)
        i++;
      }
    }
  }

  return moves;
}


const _getEndCondition = (stateHistory: any[], stateHistoryPointer: number, currentPlayer: string): ({
  handConclusion: "fold" | "showdown";
  winner: -1 | 0 | 1; // -1 for the rare event of a tie
  bestFiveCardHands?: string[];
  bestHandRankType?: string[];
}) => {
  const current_player = parseInt(currentPlayer);

  if (stateHistoryPointer >= stateHistory.length) {
    return {
      // TODO: handle tail end
      // for now, fold + tie = impossible state
      handConclusion: "fold",
      winner: -1,
      bestFiveCardHands: [],
    }
  };

  let next_prev_universal_poker_json = { acpc_state: "", best_five_card_hands: ["", ""], best_hand_rank_types: ["", ""] };

  // since the current_universal_poker_json does not contain the end move in it's history,
  // we need to go to the prev_universal_poker_json of the next one
  try {
    next_prev_universal_poker_json = JSON.parse(JSON.parse(stateHistory[stateHistoryPointer + 1]).prev_universal_poker_json);
  } catch { console.error("prev_universal_poker_json parsing failed") }

  // if the stateHistory doesn't end in a fold, it was a showdown
  const bettingString = _getMoveHistoryFromACPC(next_prev_universal_poker_json.acpc_state);

  const moves = _getMovesFromBettingStringACPC(bettingString)

  // Fold case
  if (moves.pop() === 'f') {
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
  }
}


export interface StepWithEndState {
  hand: number;
  isEndState: boolean;
  step: any;
  stateHistory: any;
  handConclusion?: "fold" | "showdown";
  winner?: -1 | 0 | 1; // -1 for the rare event of a tie
  bestFiveCardHands?: string[]; // e.g. ['AsJhTh2h2c', 'As9s9h2h2c'] (cards to be highlighted)
  bestHandRankType?: string[]; // e.g. ['High Card', 'Two Pair'] (human-readable string)
}

export const getStepsWithEndStates = (steps: any[], stateHistory: any[]): StepWithEndState[] => {
  const stepsWithEndStates: StepWithEndState[] = [];
  let handCount = 0;
  let stateHistoryPointer = 0;

  for (let i = 0; i < steps.length; i++) {

    // Find the next state history entry that is an agent action
    while (
      stateHistory[stateHistoryPointer] &&
      !_isStateHistoryAgentAction(stateHistory[stateHistoryPointer])
      && stateHistoryPointer < stateHistory.length - 1) {
      stateHistoryPointer++;
    }

    const step = steps[i];
    stepsWithEndStates.push(
      {
        hand: handCount,
        isEndState: false,
        step,
        stateHistory: stateHistory[stateHistoryPointer],
      });


    const isEndState: boolean =
      // the state history entry is at the end  
      stateHistoryPointer >= (stateHistory.length - 1) ? true
        // or the state history entry after it is an initial step
        : _isStateHistoryEntryInitial(stateHistory[stateHistoryPointer + 1]);

    if (isEndState) {

      console.log("handCount", handCount);
      console.log(step);
      const endState = _getEndCondition(stateHistory, stateHistoryPointer, step[0].observation.currentPlayer);

      // push an extra step to represent the end state
      stepsWithEndStates.push(
        {
          hand: handCount,
          isEndState: true,
          step: null,
          stateHistory: stateHistory[stateHistoryPointer],
          ...endState,
        });

      // Move to next hand
      handCount++;
    }

    stateHistoryPointer++;
  }

  return stepsWithEndStates;
}