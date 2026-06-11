import { BaseGameStep } from '@kaggle-environments/core';
import { ChessStep } from '../transformers/chessReplayTypes';
import { FORFEIT_REASONS } from '../transformers/forfeit';

export function getStepLabel(step: BaseGameStep) {
  const player = step.players.find((p) => p.isTurn);

  if (player) {
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
