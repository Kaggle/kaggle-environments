import { PokerReplay } from './poker-replay-types';

export const debugRepeatedPokerTransformer = (environment: any) => {
  const repeatedPokerReplay: PokerReplay = environment as PokerReplay;
  const repeatedPokerReplayStateHistory = environment.info.stateHistory;
  repeatedPokerReplayStateHistory.forEach((stateHistory: string) => {
    console.log(JSON.parse(stateHistory));
  });
  return repeatedPokerReplay;
};
