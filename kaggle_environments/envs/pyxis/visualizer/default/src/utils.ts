import { getStepData } from '@kaggle-environments/core';
import type {
  AssetMeta,
  AssetRow,
  AssetView,
  GameOver,
  PlayerView,
  PyxisObservation,
  RenderSnapshot,
  StepView,
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
  meta: Record<string, AssetMeta>;
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
  const index: ReplayIndex = { meta: {}, enpv: {} };
  const total = Array.isArray(replay?.steps) ? replay.steps.length : 0;
  for (let i = 0; i < total; i += 1) {
    const snap = snapshotAt(replay, i);
    if (!snap) continue;
    if (snap.meta) Object.assign(index.meta, snap.meta);
    for (const [agentId, agent] of Object.entries(snap.ag ?? {})) {
      (index.enpv[agentId] ??= []).push(agent.e);
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

function resolveAsset(row: AssetRow, meta: Record<string, AssetMeta>): AssetView {
  const [key, state, phase, timeRemaining, ptrs, level, timeOnMarket] = row;
  const [name, ta, indication, isBd, maxRevenue] = meta[key] ?? ['?', 0, 0, 0, 0];
  return {
    key,
    name,
    ta,
    indication,
    isBd: isBd === 1,
    maxRevenue,
    state,
    phase,
    timeRemaining,
    ptrs,
    level,
    timeOnMarket,
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

  const { meta, enpv } = getIndex(replay);
  const teamNames: string[] = Array.isArray(replay?.info?.TeamNames) ? replay.info.TeamNames : [];

  const players: PlayerView[] = Object.entries(snap.ag ?? {}).map(([agentId, agent], i) => ({
    name: teamNames[i] || agentId,
    agentId,
    cash: agent.c,
    enpv: agent.e,
    eroi: agent.r,
    bankrupt: agent.bk,
    operationalSites: agent.os,
    buildingSites: agent.bs,
    failed: agent.nf,
    dropped: agent.nd,
    assets: (agent.a ?? []).map((row) => resolveAsset(row, meta)),
    enpvSeries: (enpv[agentId] ?? []).slice(0, step + 1),
    reward: typeof stepData[i]?.reward === 'number' ? (stepData[i].reward as number) : null,
    status: stepData[i]?.status ?? 'ACTIVE',
  }));

  return {
    time: snap.t,
    players,
    bd: snap.bd ?? [],
    alerts: snap.al ?? [],
    markets: snap.im ?? [],
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
