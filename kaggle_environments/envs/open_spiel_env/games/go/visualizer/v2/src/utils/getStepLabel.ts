import { BaseGameStep } from '@kaggle-environments/core';
import { GoStep } from '../transformers/goReplayTypes';

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn);

  if (player) {
    const move = player.actionDisplayText?.toUpperCase() ?? '';
    if (move === 'PASS') return `Passes`;
    return `Plays on ${move}`;
  }

  const blackName = step.players.at(0)?.name ?? 'Black';
  const whiteName = step.players.at(1)?.name ?? 'White';

  // Game Start
  if (step.step === 0) {
    return `${blackName} vs. ${whiteName}`;
  }

  // Game Over
  const winner = (step as GoStep).winner;
  if (winner) {
    return `${winner === 'black' ? blackName : whiteName} wins`;
  }

  return '';
}
