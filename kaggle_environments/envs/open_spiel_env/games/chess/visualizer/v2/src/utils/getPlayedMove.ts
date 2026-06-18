import { BaseGameStep } from '@kaggle-environments/core';
import { ChessPlayer } from '../transformers/chessReplayTypes';

/**
 * Return the move string for a step, or null if no legal move was played
 * (forfeit step or no active turn). Callers can pass the result straight to
 * `chess.js`'s `Chess.move()` without an extra forfeit guard.
 */
export function getPlayedMove(step: BaseGameStep | undefined): string | null {
  const player = step?.players.find((p) => p.isTurn) as ChessPlayer | undefined;
  if (!player || player.forfeited) return null;
  return player.actionDisplayText || null;
}
