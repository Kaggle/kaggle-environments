import { defaultGetStepRenderTime } from '@kaggle-environments/core';
import type { BaseGameStep, ReplayMode } from '@kaggle-environments/core';

const ORBIT_WARS_STEP_DURATION = 440; // 2200 / 5 — RTS plays best at 5x default speed

export const getOrbitWarsStepRenderTime = (
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number
): number => {
  return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, ORBIT_WARS_STEP_DURATION);
};
