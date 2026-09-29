import { BaseGameStep, FORFEIT_REASONS } from '@kaggle-environments/core';
import { GoPlayer, GoStep } from '../transformers/goReplayTypes';

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn) as GoPlayer | undefined;

  if (player) {
    if (player.forfeited) {
      const attempted = player.forfeitLastAttempt;
      return attempted ? `Forfeited (last attempt: ${attempted})` : 'Forfeited';
    }
    const move = player.actionDisplayText?.toUpperCase() ?? '';
    if (move === 'PASS') return `Passes`;
    const hasCaptures = (step as GoStep).hasCaptures;
    return hasCaptures ? `Plays on ${move} and captures` : `Plays on ${move}`;
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
    const winnerName = winner === 'black' ? blackName : whiteName;
    const loserName = winner === 'black' ? whiteName : blackName;

    // A forfeit ends the episode before OpenSpiel reaches a terminal state, so
    // say why rather than implying the win was earned on the board.
    const status = (step as GoStep).status;
    const forfeitReason = status ? FORFEIT_REASONS[status] : undefined;
    if (forfeitReason) {
      return `${loserName} ${forfeitReason}. ${winnerName} wins by default.`;
    }
    return `${winnerName} wins`;
  }

  if ((step as GoStep).isTerminal) {
    return 'Draw';
  }

  return '';
}
