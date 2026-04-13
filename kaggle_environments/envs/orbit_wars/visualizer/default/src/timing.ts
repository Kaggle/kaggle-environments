import { defaultGetStepRenderTime } from '@kaggle-environments/core';
import type { BaseGameStep, ReplayMode } from '@kaggle-environments/core';

const ORBIT_WARS_STEP_DURATION = 550; // 2200 / 4 — RTS plays best at 4x default speed

export const getOrbitWarsStepRenderTime = (
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number
): number => {
  return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, ORBIT_WARS_STEP_DURATION);
};
