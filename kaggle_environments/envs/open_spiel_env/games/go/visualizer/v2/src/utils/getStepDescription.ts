import { BaseGameStep } from '@kaggle-environments/core';

export function getStepDescription(step: BaseGameStep) {
  const thoughts = step.players.find(p => p.isTurn)?.thoughts ?? '';
  return thoughts;
}
