/**
 * Mirror of the `render` snapshot written by `_render_snapshot` in
 * `kaggle_environments/envs/pyxis/pyxis.py`.
 *
 * Asset and market rows are tuples rather than objects: they repeat ~60x per
 * step across a 101-step replay, and naming their fields costs 16% of the whole
 * replay. The tuple elements are labeled below, so an editor names each slot on
 * hover and a destructure reads like a field access.
 */

/** Fixed order, matching `_THERAPEUTIC_AREAS` on the Python side. */
export const THERAPEUTIC_AREAS = ['Oncology', 'Respiratory & Immunology', 'Vaccines & Infectious Disease'] as const;

/** `AssetState.integer` in `game/asset.py`. */
export const ASSET_STATES = ['Idle', 'In Development', 'On Market', 'Failed', 'Expired', 'Dropped'] as const;

/** `TrialPhase.integer` in `game/trial.py`. A `-1` phase means no trial. */
export const TRIAL_PHASES = ['Phase 1', 'Phase 2', 'Phase 3', 'Approval'] as const;

/** `InvestmentLevel` in `game/constants.py`. */
export const INVESTMENT_LEVELS = ['none', 'minimal', 'standard', 'accelerated', 'stop'] as const;

/** An asset's immutable identity, sent once on the step it first appears. */
export type AssetMeta = [
  name: string,
  therapeuticArea: number,
  indication: number,
  /** 1 when acquired through business development, 0 when developed in-house. */
  isBusinessDevelopment: number,
  maxRevenue: number,
];

/** An asset's mutable state for one step. */
export type AssetRow = [
  key: string,
  state: number,
  phase: number,
  timeRemaining: number,
  /** Probability of technical and regulatory success. */
  ptrs: number,
  investmentLevel: number,
  timeOnMarket: number,
  /** GBP still owed on the running trial. Charged every step, act or not. */
  costRemaining: number,
  /** How researched `ptrs` is, 0 (one noisy reading) to 1 (sampled out). */
  ptrsEvidence: number,
  /** Accumulated brand-equity score; decays toward a floor each step. */
  brandScore: number,
];

export interface AgentSnapshot {
  /** GBP. */
  cash: number;
  /** Expected net present value, GBP. */
  enpv: number;
  /** Expected return on investment. */
  eroi: number;
  bankrupt: boolean;
  operationalSites: number;
  /** Clinical sites under construction. */
  buildingSites: number;
  failedCount: number;
  droppedCount: number;
  assets: AssetRow[];
}

export interface BdOffer {
  name: string;
  therapeuticArea: number;
  phase: number;
  maxRevenue: number;
}

export interface MarketAlert {
  /** Step the event happened on. */
  step: number;
  /** `AlertType` value: drug_release, bd_deal, pipeline_leak, ... */
  eventType: string;
  /** Agent responsible. */
  agentId: string;
  /** `-1` when the event has no area: a site auction is portfolio-wide. */
  therapeuticArea: number;
  indication: number;
  details: Record<string, unknown>;
}

export type IndicationMarket = [
  key: string,
  indicationName: string,
  firstMoverAgent: string | null,
  demandMultiplier: number,
  activeDrugCount: number,
];

export interface RenderSnapshot {
  /** Engine time, 0..horizon. */
  time: number;
  /** A clinical-site auction is taking bids this step. Opens every 20 steps. */
  siteAuctionOpen: boolean;
  agents: Record<string, AgentSnapshot>;
  bdOffers: BdOffer[];
  alerts: MarketAlert[];
  indicationMarkets: IndicationMarket[];
  /** Assets first seen this step. Absent when nothing new appeared. */
  assetMeta?: Record<string, AssetMeta>;
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
  therapeuticArea: number;
  indication: number;
  isBusinessDevelopment: boolean;
  maxRevenue: number;
  state: number;
  phase: number;
  timeRemaining: number;
  ptrs: number;
  investmentLevel: number;
  timeOnMarket: number;
  costRemaining: number;
  ptrsEvidence: number;
  brandScore: number;
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
  failedCount: number;
  droppedCount: number;
  /** Total GBP owed across every running trial. Falls due whatever happens. */
  committedCost: number;
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
  siteAuctionOpen: boolean;
  players: PlayerView[];
  bdOffers: BdOffer[];
  alerts: MarketAlert[];
  indicationMarkets: IndicationMarket[];
  gameOver: GameOver | null;
}
