import { BaseGameStep } from '@kaggle-environments/core';
import { ChessStep } from '../transformers/chessReplayTypes';

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn);

  if (player) {
    const move = player.actionDisplayText?.toUpperCase() ?? '';
    return `Plays ${move}`;
  }

  const winner = (step as ChessStep).winner;
  if (winner) {
    return winner;
  }

  return '';
}
