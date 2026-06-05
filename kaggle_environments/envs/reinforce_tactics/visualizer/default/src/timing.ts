import { defaultGetStepRenderTime } from '@kaggle-environments/core';
import type { BaseGameStep, ReplayMode } from '@kaggle-environments/core';

const REINFORCE_TACTICS_STEP_DURATION = 352; // matches kaggriculture — turn-based games can run long, keep playback snappy

export const getReinforceTacticsStepRenderTime = (
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number
): number => {
  return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, REINFORCE_TACTICS_STEP_DURATION);
};
