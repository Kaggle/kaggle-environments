import { BaseGameStep } from '@kaggle-environments/core';

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn);

  if (player === undefined && step.step === 0) {
    const black = step.players.at(0)?.name ?? 'Black';
    const white = step.players.at(1)?.name ?? 'White';
    return `${black} vs. ${white}`;
  }

  if (player === undefined) return '';

  const color = player?.id ? 'white' : 'black';
  const move = player?.actionDisplayText?.toUpperCase() ?? '';

  if (move === 'PASS') return `Played ${move}`;

  return `Played the ${color} stone on ${move}`;
}
