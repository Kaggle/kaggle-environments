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
}

export interface OpenSpielRawObservation {
  observationString?: string;
  isTerminal?: boolean;
}

export interface OpenSpielRawPlayer {
  action?: OpenSpielRawAction;
  observation?: OpenSpielRawObservation;
  reward?: number;
  status?: string;
}
