import { BaseGameStep } from '@kaggle-environments/core';

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn);
  const color = player?.id ? 'white' : 'black';
  const move = player?.actionDisplayText?.toUpperCase() ?? '';
  return `Plays a ${color} stone on ${move}`;
}
