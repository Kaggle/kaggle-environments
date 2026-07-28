// Raw step shape produced by `open_spiel_env` and consumed by every
// OpenSpiel-game visualizer transformer. Kept intentionally loose (all
// fields optional) because the env fills different subsets depending on
// the phase (setup steps, active turns, forfeit / terminal steps).

export interface OpenSpielRawAction {
  submission?: number | null;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
  /** Why the harness gave up: TRUNCATED / EMPTY / UNPARSABLE / ILLEGAL. */
  failureCategory?: string | null;
  /** One entry per LLM call the harness completed this turn, in attempt order. */
  call_details?: Array<{ finish_reason?: string | null }> | null;
}

export interface OpenSpielRawObservation {
  observationString?: string;
  isTerminal?: boolean;
}

export interface OpenSpielRawPlayer {
  action?: OpenSpielRawAction;
  observation?: OpenSpielRawObservation;
  /** null on every non-terminal step -- the env only populates rewards at the end. */
  reward?: number | null;
  status?: string;
}
