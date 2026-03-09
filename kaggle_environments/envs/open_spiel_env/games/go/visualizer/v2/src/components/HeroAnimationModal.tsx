import { memo } from 'react';
import useGameStore from '../stores/useGameStore';

export default memo(function HeroAnimationModal() {
  const game = useGameStore((state) => state.game);

  const state = game.currentState();

  const isLadder = false;
  const isPass = state.pass === true;
  const isDoublePass = state.pass === true && game._moves.at(-2).pass === true;
  const isLossOfADragon = state.capturedPositions !== undefined && state.capturedPositions.length > 6;
  const isMonkeyJump = false;

  console.log('hero animation', isLadder, isPass, isDoublePass, isLossOfADragon, isMonkeyJump);

  return null;
});
