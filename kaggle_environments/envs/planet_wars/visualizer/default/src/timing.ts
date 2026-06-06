import { defaultGetStepRenderTime } from '@kaggle-environments/core';
import type { BaseGameStep, ReplayMode } from '@kaggle-environments/core';

// 352 ms per step — matches reinforce_tactics / kaggriculture. Planet Wars
// can run up to 200 turns, so the core default of 2200 ms is far too slow.
const PLANET_WARS_STEP_DURATION = 352;

export const getPlanetWarsStepRenderTime = (
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number
): number => {
  return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, PLANET_WARS_STEP_DURATION);
};
