// Forfeit-detection helpers for OpenSpiel game visualizers.
//
// `open_spiel_env` can end an episode without OpenSpiel itself reaching a
// terminal state -- when a player exhausts their illegal-move retries, times
// out, or crashes. Renderers that only look at `observation.isTerminal` /
// `observation.winner` will freeze on the last mid-game frame with a stale
// "Turn: X" line. These helpers give per-game transformers a uniform way to
// spot the three forfeit signals and label the ending explicitly.

import { OpenSpielRawPlayer } from './types';

// Reasons the env sets on a player's top-level status when they fail to
// produce a legal action. The illegal-move path additionally routes through
// action.submission === -1 with a non-null action.status, because in that
// branch open_spiel_env normalizes both top-level statuses to DONE.
export const FORFEIT_STATUSES: ReadonlySet<string> = new Set(['TIMEOUT', 'ERROR', 'INVALID']);

export const FORFEIT_REASONS: Readonly<Record<string, string>> = {
  TIMEOUT: 'ran out of time',
  INVALID: 'submitted an illegal move',
  ERROR: 'failed to produce valid input',
};

export interface ForfeitInfo {
  index: number;
  reasonKey: string;
}

// Detect if this raw step is a forfeit by a single player. Returns the index
// of the forfeiter and the reason category (TIMEOUT / ERROR / INVALID), or
// null if not a forfeit / ambiguous (e.g. both players forfeited).
export function detectForfeit(step: OpenSpielRawPlayer[] | undefined | null): ForfeitInfo | null {
  if (!Array.isArray(step) || step.length < 2) return null;

  const statusForfeiters = step.map((p, i) => ({ p, i })).filter(({ p }) => p.status && FORFEIT_STATUSES.has(p.status));
  if (statusForfeiters.length === 1) {
    return { index: statusForfeiters[0].i, reasonKey: statusForfeiters[0].p.status! };
  }
  if (statusForfeiters.length > 1) return null;

  // illegalMoveForfeit path: env sets both top-level statuses to DONE, but
  // the offender's action carries submission=-1 with a self-reported status.
  const actionForfeiters = step
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p.action?.submission === -1 && !!p.action?.status);
  if (actionForfeiters.length === 1) {
    return { index: actionForfeiters[0].i, reasonKey: 'INVALID' };
  }
  return null;
}

// Compose the human-readable "X forfeited. Y wins by default." line.
export function buildForfeitReason(forfeit: ForfeitInfo, teamNames: string[]): string {
  const loser = teamNames[forfeit.index] ?? `Player ${forfeit.index + 1}`;
  const winner = teamNames[1 - forfeit.index] ?? `Player ${2 - forfeit.index}`;
  const reason = FORFEIT_REASONS[forfeit.reasonKey] ?? 'forfeited';
  return `${loser} ${reason}. ${winner} wins by default.`;
}
