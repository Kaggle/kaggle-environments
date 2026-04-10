import { BaseGameStep } from '@kaggle-environments/core';
import { ChessStep } from '../transformers/chessReplayTypes';

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn);

  if (player) {
    const move = player.actionDisplayText?.toUpperCase() ?? '';
    return `Plays ${move}`;
  }

  const blackName = step.players.at(0)?.name ?? 'Black';
  const whiteName = step.players.at(1)?.name ?? 'White';

  // Game Start
  if (step.step === 0) {
    return `${whiteName} vs. ${blackName}`;
  }

  // Game Over
  const winner = (step as ChessStep).winner;
  if (winner) {
    return `${winner === 'black' ? blackName : whiteName} wins`;
  }

  return '';
}
