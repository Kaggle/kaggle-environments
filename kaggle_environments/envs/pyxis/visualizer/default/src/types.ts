/**
 * Mirror of the `render` snapshot written by `_render_snapshot` in
 * `kaggle_environments/envs/pyxis/pyxis.py`.
 *
 * Keys are terse and asset rows are positional because the snapshot ships on
 * every step of a 101-step replay; see the Python docstring for the reasoning.
 */

/** Fixed order, matching `_THERAPEUTIC_AREAS` on the Python side. */
export const THERAPEUTIC_AREAS = ['Oncology', 'Respiratory & Immunology', 'Vaccines & Infectious Disease'] as const;

/** `AssetState.integer` in `game/asset.py`. */
export const ASSET_STATES = ['Idle', 'In Development', 'On Market', 'Failed', 'Expired', 'Dropped'] as const;

/** `TrialPhase.integer` in `game/trial.py`. A `-1` phase means no trial. */
export const TRIAL_PHASES = ['Phase 1', 'Phase 2', 'Phase 3', 'Approval'] as const;

/** `InvestmentLevel` in `game/constants.py`. */
export const INVESTMENT_LEVELS = ['none', 'minimal', 'standard', 'accelerated', 'stop'] as const;

/** Immutable asset identity: [name, taIndex, indication, isBd, maxRevenue]. */
export type AssetMeta = [string, number, number, number, number];

/** Mutable asset row: [key, state, phase, timeRemaining, ptrs, level, timeOnMarket]. */
export type AssetRow = [string, number, number, number, number, number, number];

export interface AgentSnapshot {
  /** Cash (GBP). */
  c: number;
  /** Expected net present value (GBP). */
  e: number;
  /** Expected return on investment. */
  r: number;
  /** Bankrupt. */
  bk: boolean;
  /** Operational clinical sites. */
  os: number;
  /** Clinical sites under construction. */
  bs: number;
  /** Failed assets. */
  nf: number;
  /** Dropped assets. */
  nd: number;
  a: AssetRow[];
}

export interface BdOffer {
  n: string;
  ta: number;
  ph: number;
  mr: number;
}

export interface MarketAlert {
  /** Step the event happened on. */
  s: number;
  /** `AlertType` value: drug_release, bd_deal, pipeline_leak, ... */
  e: string;
  /** Agent responsible. */
  a: string;
  ta: number;
  i: number;
  d: Record<string, unknown>;
}

/** [key, indicationName, firstMoverAgent, demandMultiplier, activeDrugCount]. */
export type IndicationMarket = [string, string, string | null, number, number];

export interface RenderSnapshot {
  /** Engine time, 0..horizon. */
  t: number;
  ag: Record<string, AgentSnapshot>;
  bd: BdOffer[];
  al: MarketAlert[];
  im: IndicationMarket[];
  /** Assets first seen this step. Absent when nothing new appeared. */
  meta?: Record<string, AssetMeta>;
}

export interface PyxisObservation {
  render?: RenderSnapshot;
  cash?: number;
  enpv?: number;
  bankrupt?: boolean;
}

/** One asset resolved from its meta entry plus its row for the current step. */
export interface AssetView {
  key: string;
  name: string;
  ta: number;
  indication: number;
  isBd: boolean;
  maxRevenue: number;
  state: number;
  phase: number;
  timeRemaining: number;
  ptrs: number;
  level: number;
  timeOnMarket: number;
}

export interface PlayerView {
  name: string;
  agentId: string;
  cash: number;
  enpv: number;
  eroi: number;
  bankrupt: boolean;
  operationalSites: number;
  buildingSites: number;
  failed: number;
  dropped: number;
  assets: AssetView[];
  /** eNPV at every step up to and including the current one. */
  enpvSeries: number[];
  /** Final reward, present only once the match has ended. */
  reward: number | null;
  status: string;
}

export type GameOverKind = 'horizon' | 'bankruptcy' | 'forfeit';

export interface GameOver {
  kind: GameOverKind;
  /** Index of the winner, or null for a draw. */
  winner: number | null;
  /** Index of the forfeiting player, when `kind` is 'forfeit'. */
  offender: number | null;
}

export interface StepView {
  time: number;
  players: PlayerView[];
  bd: BdOffer[];
  alerts: MarketAlert[];
  markets: IndicationMarket[];
  gameOver: GameOver | null;
}
