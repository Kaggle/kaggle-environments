import { ReplayMode, BaseGameStep, defaultGetStepRenderTime } from '@kaggle-environments/core';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import { detectHeroType } from './heroTypes';

export function getStepRenderTime(step: BaseGameStep, replayMode: ReplayMode, speedModifier: number) {
  const time = defaultGetStepRenderTime(step, replayMode, speedModifier);
  const showHeroAnimations = usePreferences.getState().showHeroAnimations;

  if (!showHeroAnimations) return time;

  const game = useGameStore.getState().game;
  const move = step?.players.find((p) => p.isTurn)?.actionDisplayText;
  const previousMove = game.history({ verbose: true }).at(0)?.san;

  if (move && move !== previousMove) game.move(move);

  if (detectHeroType(game) !== null) return 1000 * 5;

  return time;
}
