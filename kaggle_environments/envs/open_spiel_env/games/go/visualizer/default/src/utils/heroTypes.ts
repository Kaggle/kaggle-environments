import { Game } from 'tenuki';

export enum HeroTypes {
  PASS,
  DOUBLE_PASS,
  FIRST_CAPTURE,
  CRITICAL_HIT,
  DRAGON_LOSS,
}

export function detectHeroType(game: Game): HeroTypes | null {
  const state = game.currentState();

  const isPass = state.pass;
  const isDoublePass = isPass && game._moves.at(-2).pass;
  const isFirstCapture =
    state.capturedPositions?.length &&
    state.capturedPositions?.length === state.blackStonesCaptured + state.whiteStonesCaptured;
  const isCriticalHit = state.capturedPositions?.length && state.capturedPositions.length >= 10;
  const isDragonLoss = state.capturedPositions?.length && state.capturedPositions.length >= 15;

  if (isDoublePass) return HeroTypes.DOUBLE_PASS;
  if (isPass) return HeroTypes.PASS;
  if (isFirstCapture) return HeroTypes.FIRST_CAPTURE;
  if (isDragonLoss) return HeroTypes.DRAGON_LOSS;
  if (isCriticalHit) return HeroTypes.CRITICAL_HIT;

  return null;
}
