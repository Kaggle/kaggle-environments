import { defaultGetStepRenderTime } from '@kaggle-environments/core';
import type { BaseGameStep, ReplayMode } from '@kaggle-environments/core';

const KAGGRICULTURE_STEP_DURATION = 352; // 440 * 0.8 — 20% faster than orbit_wars to keep longer games snappy

export const getKaggricultureStepRenderTime = (
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number
): number => {
  return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, KAGGRICULTURE_STEP_DURATION);
};
