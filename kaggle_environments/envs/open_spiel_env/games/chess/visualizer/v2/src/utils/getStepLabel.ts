import { BaseGameStep } from '@kaggle-environments/core';
import { ChessStep } from '../transformers/chessReplayTypes';

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn);

  if (player) {
    const move = player.actionDisplayText?.toUpperCase() ?? '';
    return `Plays ${move}`;
  }

  const whiteName = step.players.at(0)?.name ?? 'White';
  const blackName = step.players.at(1)?.name ?? 'Black';

  // Game Start
  if (step.step === 0) {
    return `${blackName} vs. ${whiteName}`;
  }

  // Game Over
  const winner = (step as ChessStep).winner;
  if (winner) {
    return `${winner === 'black' ? blackName : whiteName} wins`;
  }

  return '';
}
