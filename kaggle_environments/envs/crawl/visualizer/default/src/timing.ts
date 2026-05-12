import { defaultGetStepRenderTime } from '@kaggle-environments/core';
import type { BaseGameStep, ReplayMode } from '@kaggle-environments/core';

const CRAWL_STEP_DURATION = 440; // 2200 / 5 — RTS plays best at 5x default speed

export const getCrawlStepRenderTime = (
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number
): number => {
  return defaultGetStepRenderTime(gameStep, replayMode, speedModifier, CRAWL_STEP_DURATION);
};
