import { memo } from 'react';
import { Game } from 'tenuki';
import useGameStore from '../stores/useGameStore';

export default memo(function HeroAnimationModal() {
  const game = useGameStore((state) => state.game);

  const state = game.currentState();

  function isLadder(game: Game) {
    const history = game._moves;

    if (history.length < 9) {
      return false;
    }

    if (
      history.at(-1).pass ||
      history.at(-3).pass ||
      history.at(-5).pass ||
      history.at(-7).pass ||
      history.at(-9).pass
    ) {
      return false;
    }

    const d1y = history.at(-3).playedPoint.y - history.at(-1).playedPoint.y;
    const d1x = history.at(-3).playedPoint.x - history.at(-1).playedPoint.x;
    const d2y = history.at(-5).playedPoint.y - history.at(-3).playedPoint.y;
    const d2x = history.at(-5).playedPoint.x - history.at(-3).playedPoint.x;

    if (d1y !== 0 && d1x !== 0) {
      return false;
    }

    if (d2y !== 0 && d2x !== 0) {
      return false;
    }

    if (Math.abs(d1y + d2y) !== 1 || Math.abs(d1x + d2x) !== 1) {
      return false;
    }

    if (
      history.at(-7).playedPoint.y !== history.at(-5).playedPoint.y + d1y ||
      history.at(-7).playedPoint.x !== history.at(-5).playedPoint.x + d1x
    ) {
      return false;
    }

    if (
      history.at(-9).playedPoint.y !== history.at(-7).playedPoint.y + d2y ||
      history.at(-9).playedPoint.x !== history.at(-7).playedPoint.x + d2x
    ) {
      return false;
    }

    return true;
  }

  const isPass = state.pass === true;
  const isDoublePass = state.pass === true && game._moves.at(-2).pass === true;
  const isLossOfADragon = state.capturedPositions !== undefined && state.capturedPositions.length > 6;
  const isMonkeyJump = false;

  console.log('hero animation', isLadder(game), isPass, isDoublePass, isLossOfADragon, isMonkeyJump);

  return null;
});
