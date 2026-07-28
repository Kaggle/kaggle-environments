/* eslint-disable @typescript-eslint/no-explicit-any */
import { detectForfeit } from '@kaggle-environments/core';
import { ChessAttempt, ChessReplay, ChessPlayer, ChessStep, FenState, ChessReplayStep } from './chessReplayTypes';

function parseFen(fen?: string): FenState {
  if (!fen || typeof fen !== 'string') {
    return {
      board: [],
      activeColor: '',
      castling: '',
      enPassant: '',
      halfmoveClock: '',
      fullmoveNumber: '',
    };
  }

  const [piecePlacement, activeColor, castling, enPassant, halfmoveClock, fullmoveNumber] = fen.split(' ');

  // Within the context of the replay, the active color is the color of the player that just completed their move.
  // This is the opposite of the active color in the fen string, which is the color of the player that is about to move.
  // Therefore, we need to invert the "active" color.
  const playerColor = String(activeColor).toLowerCase() === 'w' ? 'Black' : 'White';

  const board = [];
  const rows = piecePlacement.split('/');

  for (const row of rows) {
    const boardRow = [];
    for (const char of row) {
      if (isNaN(parseInt(char))) {
        boardRow.push(char);
      } else {
        for (let i = 0; i < parseInt(char); i++) {
          boardRow.push(null);
        }
      }
    }
    board.push(boardRow);
  }

  return {
    board,
    activeColor: playerColor,
    castling,
    enPassant,
    halfmoveClock,
    fullmoveNumber,
  };
}

export function getChessStepDescription(step: ChessStep) {
  if (step.isTerminal) {
    return '';
  }

  const player = step.players.find((p) => p.isTurn);
  if (!player) return '';
  return renderAttemptsMarkdown(player);
}

/**
 * Render a player's per-attempt LLM calls as markdown. When there's only one
 * attempt this collapses to just the response (the legacy behavior). When
 * there are retries each attempt gets a header showing its outcome:
 *   - intermediate attempts → ❌ Attempt N (illegal — retried)
 *   - final attempt on a successful turn → ✅ Attempt N (submitted)
 *   - all attempts on a forfeit → ❌ Attempt N (illegal — forfeited on last)
 *
 * Falls back to player.thoughts if call_details aren't available (older
 * replays from before the harness wrote call_details).
 */
function renderAttemptsMarkdown(player: ChessPlayer): string {
  const attempts = player.attempts ?? [];
  const fallback = player.thoughts ?? '';

  if (attempts.length === 0) return fallback;

  if (attempts.length === 1 && !player.forfeited) {
    // Single legal attempt — keep the original clean rendering.
    return attempts[0].response || fallback;
  }

  const total = attempts.length;
  const lines: string[] = [];

  if (player.forfeited) {
    const lastMove = player.forfeitLastAttempt ? ` \`${player.forfeitLastAttempt}\`` : '';
    lines.push(`⚠️ **Forfeited after ${total} attempt${total === 1 ? '' : 's'}.** Last attempt:${lastMove}`);
    lines.push('');
  } else {
    lines.push(`🔁 **Took ${total} attempts** to find a legal move.`);
    lines.push('');
  }

  attempts.forEach((attempt, i) => {
    const isLast = i === attempts.length - 1;
    const ok = isLast && !player.forfeited;
    const outcome = isLast ? 'forfeited' : 'retried';
    const cause = attempt.finishReason === 'length' ? 'cut off at token limit' : 'illegal';
    const tag = ok
      ? `✅ **Attempt ${i + 1} of ${total}** (submitted)`
      : `❌ **Attempt ${i + 1} of ${total}** (${cause} — ${outcome})`;
    lines.push(`### ${tag}`);
    lines.push('');
    lines.push(attempt.response || '_(empty response)_');
    lines.push('');
  });

  return lines.join('\n').trim();
}

function deriveWinner(step: ChessReplayStep[]): string | null {
  if (step[0].reward === 1) return 'black';
  if (step[1].reward === 1) return 'white';
  return null;
}

export const chessTransformer = (environment: any): ChessStep[] => {
  const chessReplay = environment as ChessReplay;
  const chessSteps: ChessStep[] = [];

  const extraStepPlayers = [0, 1].map(
    (index): ChessPlayer => ({
      id: index,
      name: environment.info.TeamNames[index],
      thumbnail: '',
      isTurn: false,
      actionDisplayText: '',
      thoughts: '',
      reward: null,
      generateReturns: null,
    })
  );

  chessSteps.push({
    step: chessSteps.length,
    players: extraStepPlayers,
    fenState: parseFen(''),
    isTerminal: false,
    winner: null,
    status: null,
  });

  for (const step of chessReplay.steps) {
    // Each step contains a tuple of players, one who acted and one who's waiting
    const stepPlayers: ChessPlayer[] = step.map((player, index): ChessPlayer => {
      const attempts: ChessAttempt[] =
        player.action?.call_details?.map((c) => ({
          response: c.response ?? '',
          finishReason: c.finish_reason ?? null,
        })) ?? [];
      // A forfeit step is one where the player submitted -1 *and* we have a
      // non-null action.status. Inactive turns also have submission === -1
      // but with null action.status.
      const submission = player.action?.submission;
      const forfeited = submission === -1 && !!player.action?.status;
      return {
        id: index,
        name: environment.info.TeamNames[index],
        thumbnail: '',
        // A turn requires submission to be a real action id. -1 means the player
        // didn't act this step (inactive or forfeited). null/undefined shows up
        // in init steps, we don't need those rendered. Treat forfeits as a
        // "turn" too so the step is preserved.
        isTurn: (typeof submission === 'number' && submission !== -1) || forfeited,
        // Raw move only — chess.js consumes this directly. Forfeit decoration
        // happens at display sites (getStepLabel) using the `forfeited` flag.
        actionDisplayText: player.action?.actionString ?? '',
        thoughts: player.action?.thoughts ?? '',
        reward: player.reward,
        generateReturns: player.action?.generate_returns ?? null,
        attempts,
        forfeited,
        forfeitLastAttempt: forfeited ? (player.action?.actionString ?? null) : null,
      };
    });

    // Ignore setup steps where no one acted
    if (stepPlayers.findIndex((player) => player.isTurn) !== -1) {
      chessSteps.push({
        step: chessSteps.length,
        players: stepPlayers,
        // Both agents have the same observation string for the step, just grab the first one
        fenState: parseFen(step[0].observation.observationString),
        isTerminal: false,
        winner: '',
        status: null,
      });
    }
  }

  const lastReplayStep = chessReplay.steps[chessReplay.steps.length - 1];

  chessSteps.push({
    players: extraStepPlayers,
    isTerminal: true,
    fenState: chessSteps[chessSteps.length - 1].fenState,
    step: chessSteps.length,
    winner: deriveWinner(lastReplayStep),
    // Only the reason category is rendered here — GameOver/getStepLabel derive
    // the loser from the winner, so detectForfeit's index isn't needed.
    status: detectForfeit(lastReplayStep)?.reasonKey ?? null,
  });

  return chessSteps;
};
