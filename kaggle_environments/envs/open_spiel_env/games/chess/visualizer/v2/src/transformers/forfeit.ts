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
 * the offender. The transformer's deriveStatus() handles that case.
 */
export const FORFEIT_STATUSES = new Set(['TIMEOUT', 'ERROR', 'INVALID']);

/**
 * Human-readable phrasing for each forfeit status. Slots into "{loser} {reason}."
 * Phrases match the canonical mapping in the default chess visualizer so the
 * scrubber label and game-over modal stay consistent.
 */
export const FORFEIT_REASONS: Record<string, string> = {
  TIMEOUT: 'ran out of time',
  INVALID: 'submitted an illegal move',
  ERROR: 'failed to produce valid input',
};
