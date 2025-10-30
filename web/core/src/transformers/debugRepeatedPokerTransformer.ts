import { PokerReplay, PokerReplayStepHistoryParsed } from './poker-replay-types';

export const debugRepeatedPokerTransformer = (environment: any) => {
  const repeatedPokerReplay: PokerReplay = environment as PokerReplay;
  const repeatedPokerReplayStateHistory = environment.info.stateHistory;
  const parsedStateHistorySteps = repeatedPokerReplayStateHistory.map((stateHistory: string) => {
    const stateHistoryObject = JSON.parse(stateHistory);
    stateHistoryObject['current_universal_poker_json'] = JSON.parse(stateHistoryObject['current_universal_poker_json']);
    if (stateHistoryObject['prev_universal_poker_json']) {
      stateHistoryObject['prev_universal_poker_json'] = JSON.parse(stateHistoryObject['prev_universal_poker_json']);
    }
    return stateHistoryObject;
  });
  const onlyPlayerSteps = repeatedPokerReplay.steps.flat().filter((step) => step.action.submission > -1);

  let stepToMapIndex = 0;
  const stateHistoryStepsWithReplaySteps = parsedStateHistorySteps.map(
    (parsedStateHistoryStep: PokerReplayStepHistoryParsed) => {
      if (parsedStateHistoryStep.current_universal_poker_json.current_player > -1) {
        parsedStateHistoryStep.step = onlyPlayerSteps[stepToMapIndex];
        parsedStateHistoryStep.stepIndex = stepToMapIndex;
        stepToMapIndex++;
      }
      return parsedStateHistoryStep;
    }
  );

  return stateHistoryStepsWithReplaySteps;
};
