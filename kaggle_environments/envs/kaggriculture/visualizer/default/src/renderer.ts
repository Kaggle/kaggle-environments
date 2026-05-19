import type { RendererOptions } from '@kaggle-environments/core';
import { getStepData } from '@kaggle-environments/core';
import { buildShell, collectRefs, renderObservation, type BoardSize, type LayoutRefs } from './renderFarm';
import { MARKET_ITEMS, type ViewModel, type PrivateState } from './types';

const DEFAULT_BOARD: BoardSize = { rows: 10, cols: 10 };

interface CachedShell {
  key: string;
  refs: LayoutRefs;
}

const shellCache = new WeakMap<HTMLElement, CachedShell>();

function inferBoardSize(replay: any): BoardSize {
  const cfg = replay?.configuration ?? {};
  const size = Number(cfg.boardSize);
  if (Number.isFinite(size) && size > 0) {
    return { rows: size, cols: size };
  }
  return DEFAULT_BOARD;
}

function pricesAt(replay: any, step: number): Record<string, number> {
  const data = getStepData(replay, step) as any;
  if (!data) return {};
  const entry = Array.isArray(data) ? data[0] : data;
  return entry?.observation?.market?.prices ?? {};
}

function buildPriceHistory(replay: any, step: number, windowSize: number): Record<string, number[]> {
  const startPrices = pricesAt(replay, 0);
  const firstStep = Math.max(0, step - windowSize + 1);
  const windowPrices: Record<string, number>[] = [];
  for (let s = firstStep; s <= step; s++) windowPrices.push(pricesAt(replay, s));
  const padCount = windowSize - windowPrices.length;
  const history: Record<string, number[]> = {};
  for (const { key } of MARKET_ITEMS) {
    const start = Number(startPrices[key] ?? 0);
    const series: number[] = new Array(padCount).fill(start);
    for (const p of windowPrices) {
      const v = p[key];
      series.push(Number(v == null ? start : v));
    }
    history[key] = series;
  }
  return history;
}

function buildView(replay: any, step: number, turnsPerDay: number): ViewModel | null {
  const stepData = getStepData(replay, step) as any;
  if (!stepData) return null;
  const entries = Array.isArray(stepData) ? stepData : [stepData];
  const obs0 = entries[0]?.observation;
  if (!obs0) return null;

  const privates: (PrivateState | undefined)[] = [];
  for (const entry of entries) {
    const obs = entry?.observation;
    const idx = typeof obs?.player === 'number' ? obs.player : privates.length;
    privates[idx] = obs?.private;
  }

  return {
    day: Number(obs0.day ?? 0),
    hour: Number(obs0.hour ?? 0),
    farms: obs0.farms ?? [],
    market: obs0.market ?? { prices: {}, inventory: {} },
    town: obs0.town ?? { unlocked_shops: [] },
    privates,
    priceHistory: buildPriceHistory(replay, step, turnsPerDay),
  };
}

export function renderer(options: RendererOptions): void {
  const { parent, replay, step, agents } = options;
  if (!parent || !replay) return;

  const board = inferBoardSize(replay);
  const playerNames = [0, 1].map((i) => agents?.[i]?.name || `Player ${i + 1}`);
  const expectedKey = `${board.rows}x${board.cols}|${playerNames.join('|')}`;
  let cached = shellCache.get(parent);
  if (!cached || cached.key !== expectedKey) {
    buildShell(parent, board, playerNames);
    cached = { key: expectedKey, refs: collectRefs(parent, board) };
    shellCache.set(parent, cached);
  }

  const cfg = replay?.configuration ?? {};
  const turnsPerDay = Math.max(1, Number(cfg.turnsPerDay) || 24);
  const view = buildView(replay, step, turnsPerDay);
  if (!view) return;
  renderObservation(cached.refs, view, cfg);
}
