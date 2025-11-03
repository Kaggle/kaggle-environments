import { PokerReplay, PokerReplayStepHistoryParsed } from './poker-replay-types';

import { createVisualStepsFromRepeatedPokerReplay } from './repeatedPokerTransformerUtils';

export const repeatedPokerTransformerV2 = (environment: any) => {
  const repeatedPokerReplay: PokerReplay = environment as PokerReplay;
  const repeatedPokerReplayStateHistory = environment.info.stateHistory;
  const agents = environment.info.Agents;

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
  const stateHistoryStepsWithReplaySteps: PokerReplayStepHistoryParsed[] = parsedStateHistorySteps.map(
    (parsedStateHistoryStep: PokerReplayStepHistoryParsed) => {
      if (parsedStateHistoryStep.current_universal_poker_json.current_player > -1) {
        parsedStateHistoryStep.step = onlyPlayerSteps[stepToMapIndex];
        parsedStateHistoryStep.stepIndex = stepToMapIndex;
        stepToMapIndex++;
      }
      return parsedStateHistoryStep;
    }
  );

  // this is a thing you can comment out to see the input/output for the first full game + 1 step in the input
  // console.log(stateHistoryStepsWithReplaySteps.slice(0,13));
  // console.log(createVisualStepsFromRepeatedPokerReplay(stateHistoryStepsWithReplaySteps.slice(0,13), agents));

  return createVisualStepsFromRepeatedPokerReplay(stateHistoryStepsWithReplaySteps, agents);

};
