import { ReplayData } from '@kaggle-environments/core';
import { goTransformer } from './goTransformer';
import { GoStep } from './goReplayTypes';

export function transformer(replay: ReplayData): ReplayData<GoStep[]> {
  return {
    ...replay,
    steps: goTransformer(replay),
    isTransformed: true,
  };
}
