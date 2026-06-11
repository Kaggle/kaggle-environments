import { BaseGameStep } from '@kaggle-environments/core';
import { ChessPlayer, ChessStep } from '../transformers/chessReplayTypes';

const FORFEIT_REASONS: Record<string, string> = {
  TIMEOUT: 'ran out of time',
  INVALID: 'submitted an illegal move',
  ERROR: 'failed to produce valid input',
};

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn) as ChessPlayer | undefined;

  if (player) {
    if (player.forfeited) {
      return player.forfeitLastAttempt ? `Forfeits (attempted ${player.forfeitLastAttempt})` : 'Forfeits';
    }
    const move = player.actionDisplayText ?? '';
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
    const winnerName = winner === 'black' ? blackName : whiteName;
    const loserName = winner === 'black' ? whiteName : blackName;
    const baseLabel = `${winnerName} wins`;
    const status = (step as ChessStep).status;
    const reason = status ? FORFEIT_REASONS[status] : undefined;
    if (reason) {
      return `${baseLabel}\n${loserName} ${reason}. ${winnerName} wins by default.`;
    }
    return baseLabel;
  }

  if ((step as ChessStep).isTerminal) {
    return 'Draw';
  }

  return '';
}
