
const _isStateHistoryAgentAction = (stateHistoryEntry: string): boolean =>
  JSON.parse(JSON.parse(stateHistoryEntry).current_universal_poker_json).current_player !== -1;

const _isStateHistoryEntryInitial = (stateHistoryEntry: string): boolean => {
  const state = JSON.parse(JSON.parse(stateHistoryEntry).current_universal_poker_json);
  return state.acpc_state.startsWith('STATE:0::2c2c|2c2c');
}

export interface StepWithEndState {
  hand: number;
  isEndState: boolean;
  step: any;
  stateHistory: any;
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
      && stateHistoryPointer < stateHistory.length) {
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
      // push an extra step to represent the end state
      stepsWithEndStates.push(
        {
          hand: handCount,
          isEndState: true,
          step: null,
          stateHistory: stateHistory[stateHistoryPointer],
        });
      handCount++;
    }

    stateHistoryPointer++;
  }

  return stepsWithEndStates;
}