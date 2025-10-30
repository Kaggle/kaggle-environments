import { PokerReplay } from './poker-replay-types';

export const debugRepeatedPokerTransformer = (environment: any) => {
  const repeatedPokerReplay: PokerReplay = environment as PokerReplay;
  const repeatedPokerReplayStateHistory = environment.info.stateHistory;
  const parsedStateHistorySteps = repeatedPokerReplayStateHistory.map((stateHistory: string) => {
    const stateHistoryObject = JSON.parse(stateHistory);
    stateHistoryObject['current_universal_poker_json'] = JSON.parse(stateHistoryObject['current_universal_poker_json']);
    return stateHistoryObject;
  });
  return repeatedPokerReplay.steps;

  //return parsedStateHistorySteps;
};
