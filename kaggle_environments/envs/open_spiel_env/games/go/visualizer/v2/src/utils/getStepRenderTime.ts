import { ReplayMode, BaseGameStep, defaultGetStepRenderTime } from '@kaggle-environments/core';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import { detectHeroType } from '../utils/heroTypes';

export function getStepRenderTime(step: BaseGameStep, replayMode: ReplayMode, speedModifier: number) {
  const time = defaultGetStepRenderTime(step, replayMode, speedModifier);
  const showHeroAnimations = usePreferences.getState().showHeroAnimations;
  const reducedMotion = usePreferences.getState().reducedMotion;

  if (reducedMotion || !showHeroAnimations) return time * 1.6; // Default of approx 3.5s

  const game = useGameStore.getState().game;
  // Temporary hack: The step render time is calculated before the step
  // is rendered, so make the move for the most recent step before
  // working out if it's playback duration needs adjusting.
  const move = step?.players.find((p) => p.isTurn)?.actionDisplayText;

  if (move === 'PASS') {
    game.pass();
  } else if (move) {
    const y = game.boardSize - parseInt(move.slice(1));
    const x = 'abcdefghjklmnopqrst'.indexOf(move.charAt(0));
    game.playAt(y, x);
  }

  if (detectHeroType(game) !== null) return 1000 * 5;

  return time * 1.6; // Default of approx 3.5s;
}
