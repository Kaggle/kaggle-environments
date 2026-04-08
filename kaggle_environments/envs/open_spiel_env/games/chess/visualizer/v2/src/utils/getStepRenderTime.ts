import { ReplayMode, BaseGameStep, defaultGetStepRenderTime } from '@kaggle-environments/core';

export function getStepRenderTime(step: BaseGameStep, replayMode: ReplayMode, speedModifier: number) {
  const time = defaultGetStepRenderTime(step, replayMode, speedModifier);
  return time;
}
