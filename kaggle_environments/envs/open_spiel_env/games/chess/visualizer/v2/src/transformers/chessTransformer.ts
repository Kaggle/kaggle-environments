/* eslint-disable @typescript-eslint/no-explicit-any */
import { ChessAttempt, ChessReplay, ChessPlayer, ChessStep, FenState, ChessReplayStep } from './chessReplayTypes';
import { FORFEIT_STATUSES } from './forfeit';

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
      const attempts: ChessAttempt[] = player.action?.call_details?.map((c) => ({ response: c.response ?? '' })) ?? [];
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
        // in init steps, we don't need those rendered.
        isTurn: typeof submission === 'number' && submission !== -1,
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
    status: deriveStatus(lastReplayStep),
  });

  return chessSteps;
};
