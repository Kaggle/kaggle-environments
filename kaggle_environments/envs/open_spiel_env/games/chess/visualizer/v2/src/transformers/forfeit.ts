/**
 * Statuses set by open_spiel_env when an agent fails to produce a valid action:
 *   TIMEOUT — exceeded the per-move / overage time budget
 *   ERROR   — agent crashed or response was unparsable / cut off
 *   INVALID — submitted an illegal move
 *
 * Note: forfeit is only visible via action.submission === -1 + a non-null
 * action.status on the offender. deriveStatus() is responsible for detecting that.
 */
export const FORFEIT_STATUSES = new Set(['TIMEOUT', 'ERROR', 'INVALID']);

/**
 * Slots into "{loser} {reason}."
 */
export const FORFEIT_REASONS: Record<string, string> = {
  TIMEOUT: 'ran out of time',
  INVALID: 'submitted an illegal move',
  ERROR: 'failed to produce valid input',
};
