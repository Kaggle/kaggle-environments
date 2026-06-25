import { ReplayData } from '@kaggle-environments/core';
import { chessTransformer } from './chessTransformer';
import { ChessStep } from './chessReplayTypes';

export function transformer(replay: ReplayData): ReplayData<ChessStep[]> {
  return {
    ...replay,
    steps: chessTransformer(replay),
    isTransformed: true,
  };
}
