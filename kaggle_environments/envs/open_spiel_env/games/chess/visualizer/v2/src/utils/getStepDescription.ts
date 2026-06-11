import { BaseGameStep } from '@kaggle-environments/core';
import { ChessPlayer } from '../transformers/chessReplayTypes';
import { renderAttemptsMarkdown } from '../transformers/chessTransformer';

export function getStepDescription(step: BaseGameStep): string {
  const player = step.players.find((p) => p.isTurn) as ChessPlayer | undefined;
  if (!player) return '';
  return renderAttemptsMarkdown(player);
}
