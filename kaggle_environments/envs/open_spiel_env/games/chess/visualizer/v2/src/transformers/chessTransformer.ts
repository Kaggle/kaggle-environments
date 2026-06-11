/* eslint-disable @typescript-eslint/no-explicit-any */
import { ChessAttempt, ChessReplay, ChessPlayer, ChessStep, FenState, ChessReplayStep } from './chessReplayTypes';

/**
 * Statuses set by open_spiel_env when an agent fails to produce a valid action:
 *   TIMEOUT — exceeded the per-move / overage time budget
 *   ERROR   — agent crashed or response was unparsable / cut off
 *   INVALID — submitted an illegal move
 * In all three cases the opponent wins by default.
 *
 * Note: when illegalMoveForfeit:true and the env's INVALID branch runs, both
 * players' top-level status gets overwritten to DONE — the per-player forfeit
 * is only visible via action.submission === -1 + a non-null action.status on
 * the offender. deriveStatus() below handles that case.
 */
const FORFEIT_STATUSES = new Set(['TIMEOUT', 'ERROR', 'INVALID']);

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

function deriveWinner(step: ChessReplayStep[]): string | null {
  if (step[0].reward === 1) return 'black';
  if (step[1].reward === 1) return 'white';
  return null;
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
export function renderAttemptsMarkdown(player: ChessPlayer): string {
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
    lines.push(`> 🔁 **Took ${total} attempts** to find a legal move.`);
    lines.push('');
  }

  attempts.forEach((attempt, i) => {
    const isLast = i === attempts.length - 1;
    const ok = isLast && !player.forfeited;
    const tag = ok
      ? `✅ **Attempt ${i + 1} of ${total}** (submitted)`
      : `❌ **Attempt ${i + 1} of ${total}** (illegal — ${isLast ? 'forfeited' : 'retried'})`;
    lines.push(`### ${tag}`);
    lines.push('');
    lines.push(attempt.response || '_(empty response)_');
    lines.push('');
  });

  return lines.join('\n').trim();
}

/**
 * Detect the forfeit status from the terminal replay step, or null if no
 * single player unambiguously forfeited.
 *
 * Two signals:
 *   1. Top-level player.status in FORFEIT_STATUSES — used in strict mode
 *      and for ERROR/TIMEOUT cases where the env doesn't overwrite status.
 *   2. action.submission === -1 with a non-null action.status — used for
 *      the illegalMoveForfeit:true path, where the env normalizes both
 *      top-level statuses to DONE but leaves the offender's self-reported
 *      forfeit message on action.status.
 *
 * Returns null when no detector fires OR when both players match (genuinely
 * undetermined — episode voided).
 */
function deriveStatus(step: ChessReplayStep[]): string | null {
  if (step.length < 2) return null;

  const statusForfeiters = step.filter((p) => FORFEIT_STATUSES.has(p.status));
  if (statusForfeiters.length === 1) return statusForfeiters[0].status;
  if (statusForfeiters.length > 1) return null;

  const actionForfeiters = step.filter((p) => p.action?.submission === -1 && p.action?.status);
  if (actionForfeiters.length === 1) {
    // Reuse INVALID phrasing — submission=-1 with action.status is the same
    // forfeit-by-illegal-move mechanism, routed through the env's
    // invalid_action branch (which normalizes top-level status to DONE).
    return 'INVALID';
  }

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
      const submission = player.action?.submission;
      const attempts: ChessAttempt[] = player.action?.call_details?.map((c) => ({ response: c.response ?? '' })) ?? [];
      // A forfeit step is one where the player submitted -1 *and* the harness
      // wrote a self-reported status (action.status). Inactive turns also
      // have submission === -1 but with null action.status.
      const forfeited = submission === -1 && !!player.action?.status;
      return {
        id: index,
        name: environment.info.TeamNames[index],
        thumbnail: '',
        // Treat forfeits as the player's "turn" too so the step survives the
        // setup-step filter below and the slider lands on the moment of forfeit.
        isTurn: submission !== -1 || forfeited,
        // Keep actionDisplayText empty for forfeits — actionString here is the
        // rejected (illegal) move, and GameRenderer/getStepRenderTime feed this
        // into chess.js. The rejected move is surfaced via forfeitLastAttempt
        // for label/UI use instead.
        actionDisplayText: forfeited ? '' : (player.action?.actionString ?? ''),
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
    status: deriveStatus(lastReplayStep),
  });

  return chessSteps;
};
