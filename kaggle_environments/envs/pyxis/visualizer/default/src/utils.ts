import { getStepData } from '@kaggle-environments/core';
import {
  ASSET_STATES,
  type AgentSnapshot,
  type AssetMeta,
  type AssetRow,
  type AssetView,
  type GameOver,
  type Moves,
  type PlayerView,
  type PyxisObservation,
  type RenderSnapshot,
  type StepView,
} from './types';

/**
 * Per-replay derived data that is expensive to recompute on every scrub.
 *
 * Asset identity is emitted only on the step an asset first appears, so
 * resolving a row at step N means having walked steps 0..N. Both that index and
 * the eNPV series are built once per replay and cached against the replay
 * object itself.
 */
interface ReplayIndex {
  assetMeta: Record<string, AssetMeta>;
  enpv: Record<string, number[]>;
}

const indexCache = new WeakMap<object, ReplayIndex>();

function snapshotAt(replay: any, step: number): RenderSnapshot | null {
  const stepData = getStepData<PyxisObservation>(replay, step);
  if (!stepData) return null;
  // `render` is a `shared` property, so it lives on seat 0 only.
  return stepData[0]?.observation?.render ?? null;
}

function buildIndex(replay: any): ReplayIndex {
  const index: ReplayIndex = { assetMeta: {}, enpv: {} };
  const total = Array.isArray(replay?.steps) ? replay.steps.length : 0;
  for (let i = 0; i < total; i += 1) {
    const snap = snapshotAt(replay, i);
    if (!snap) continue;
    if (snap.assetMeta) Object.assign(index.assetMeta, snap.assetMeta);
    for (const [agentId, agent] of Object.entries(snap.agents ?? {})) {
      (index.enpv[agentId] ??= []).push(agent.enpv);
    }
  }
  return index;
}

function getIndex(replay: any): ReplayIndex {
  let cached = indexCache.get(replay);
  if (!cached) {
    cached = buildIndex(replay);
    indexCache.set(replay, cached);
  }
  return cached;
}

function resolveAsset(row: AssetRow, assetMeta: Record<string, AssetMeta>): AssetView {
  const [key, state, phase, timeRemaining, ptrs, investmentLevel, timeOnMarket, ptrsReadings, brandLift, patentLeft] =
    row;
  const [name, therapeuticArea, indication, isBusinessDevelopment, maxRevenue] = assetMeta[key] ?? ['?', 0, 0, 0, 0];
  return {
    key,
    name,
    therapeuticArea,
    indication,
    isBusinessDevelopment: isBusinessDevelopment === 1,
    maxRevenue,
    state,
    phase,
    timeRemaining,
    ptrs,
    investmentLevel,
    timeOnMarket,
    ptrsReadings,
    brandLift,
    patentLeft,
  };
}

/** Count of a numeric action head's entries matching `test`. */
function countWhere(values: unknown, test: (v: number) => boolean): number {
  return Array.isArray(values) ? values.filter((v) => typeof v === 'number' && test(v)).length : 0;
}

/**
 * What a player did on the step just played.
 *
 * Bids, readings and marketing come straight from the submitted action. Trial
 * starts and launches come from diffing asset states against the previous
 * step instead: the `investments` head is indexed by an asset slot order the
 * replay does not record, and an invest on a running trial is a no-op anyway.
 */
function movesOf(action: any, agent: AgentSnapshot, prev: AgentSnapshot | undefined): Moves {
  // A bankrupt portfolio is frozen; whatever it still submits is ignored.
  const frozen = prev?.bankrupt ?? false;
  const a = action && typeof action === 'object' && !frozen ? action : {};
  // Bids are GBP millions; one that rounds to zero is a pass.
  const bids: number[] = Array.isArray(a.bd_bids) ? a.bd_bids.filter((b: unknown) => typeof b === 'number') : [];
  const placed = bids.filter((b) => Math.round(b) > 0);
  const siteBid = Array.isArray(a.site_bid) ? Number(a.site_bid[0]) || 0 : 0;

  const prevState = new Map((prev?.assets ?? []).map((row) => [row[0], ASSET_STATES[row[1]]]));
  let started = 0;
  let launched = 0;
  for (const [key, state] of agent.assets) {
    const before = prevState.get(key);
    const now = ASSET_STATES[state];
    if (before === 'Idle' && now === 'In Development') started += 1;
    if (before === 'In Development' && now === 'On Market') launched += 1;
  }

  return {
    frozen,
    started,
    launched,
    failed: prev ? agent.failedCount - prev.failedCount : 0,
    dropped: prev ? agent.droppedCount - prev.droppedCount : 0,
    readings: Array.isArray(a.ptrs_research)
      ? a.ptrs_research.reduce((sum: number, n: unknown) => sum + (typeof n === 'number' ? n : 0), 0)
      : 0,
    bdBid: placed.reduce((sum, b) => sum + b, 0) * 1e6,
    bdBids: placed.length,
    siteBid: Math.round(siteBid) > 0 ? siteBid * 1e6 : 0,
    boughtSite: Number(a.upgrade) > 0,
    brandEquity: countWhere(a.brand_equity, (v) => v > 0),
    demandCreation: countWhere(a.demand_creation, (v) => v > 0),
  };
}

/**
 * Classify the end of the match from the final step's statuses and rewards.
 *
 * Returns null while the match is still running, so the renderer can tell
 * "mid-game" from "ended in a draw" rather than freezing on the last frame.
 */
function detectGameOver(entries: any[], players: PlayerView[]): GameOver | null {
  const offender = entries.findIndex(
    (e) => e?.status === 'INVALID' || e?.status === 'ERROR' || e?.status === 'TIMEOUT'
  );
  if (offender >= 0) {
    const winner = entries.findIndex((e, i) => i !== offender && e?.reward === 1.0);
    return { kind: 'forfeit', winner: winner >= 0 ? winner : null, offender };
  }

  const done = entries.length > 0 && entries.every((e) => e?.status === 'DONE');
  if (!done) return null;

  const rewards = entries.map((e) => (typeof e?.reward === 'number' ? e.reward : null));
  const best = Math.max(...rewards.map((r) => r ?? -Infinity));
  const winners = rewards.map((r, i) => (r === best ? i : -1)).filter((i) => i >= 0);
  const winner = winners.length === 1 ? winners[0] : null;
  const kind = players.some((p) => p.bankrupt) ? 'bankruptcy' : 'horizon';
  return { kind, winner, offender: null };
}

export function buildView(replay: any, step: number): StepView | null {
  const stepData = getStepData<PyxisObservation>(replay, step);
  if (!stepData) return null;
  const snap = snapshotAt(replay, step);
  if (!snap) return null;

  const { assetMeta, enpv } = getIndex(replay);
  const prevSnap = step > 0 ? snapshotAt(replay, step - 1) : null;
  const teamNames: string[] = Array.isArray(replay?.info?.TeamNames) ? replay.info.TeamNames : [];

  const players: PlayerView[] = Object.entries(snap.agents ?? {}).map(([agentId, agent], i) => {
    const assets = (agent.assets ?? []).map((row) => resolveAsset(row, assetMeta));
    return {
      name: teamNames[i] || agentId,
      agentId,
      cash: agent.cash,
      enpv: agent.enpv,
      eroi: agent.eroi,
      bankrupt: agent.bankrupt,
      operationalSites: agent.operationalSites,
      buildingSites: agent.buildingSites,
      failedCount: agent.failedCount,
      droppedCount: agent.droppedCount,
      freeSites: agent.freeSites,
      trialBurn: agent.trialBurn,
      committedCost: agent.committedCost,
      revenue: agent.revenue,
      spend: agent.spend,
      endedReason: agent.endedReason,
      moves: movesOf(stepData[i]?.action, agent, prevSnap?.agents?.[agentId]),
      assets,
      enpvSeries: (enpv[agentId] ?? []).slice(0, step + 1),
      reward: typeof stepData[i]?.reward === 'number' ? (stepData[i].reward as number) : null,
      status: stepData[i]?.status ?? 'ACTIVE',
    };
  });

  return {
    time: snap.time,
    siteAuctionOpen: snap.siteAuctionOpen ?? false,
    players,
    bdOffers: snap.bdOffers ?? [],
    alerts: snap.alerts ?? [],
    indicationMarkets: snap.indicationMarkets ?? [],
    gameOver: detectGameOver(stepData as any[], players),
  };
}

/** Compact GBP formatting: 7_400_000_000 -> "£7.4B". */
export function formatMoney(value: number): string {
  const sign = value < 0 ? '-' : '';
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${sign}£${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}£${(abs / 1e6).toFixed(0)}M`;
  if (abs >= 1e3) return `${sign}£${(abs / 1e3).toFixed(0)}K`;
  return `${sign}£${abs.toFixed(0)}`;
}
