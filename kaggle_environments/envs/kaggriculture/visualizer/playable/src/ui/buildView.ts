/**
 * Bridges live GameState → the default visualizer's ViewModel shape.
 *
 * Default builds a window of recent prices from the replay. We don't have a
 * replay here; we accumulate a rolling per-item price history across steps
 * outside React state and expose `record(state)` to push the latest sample.
 */

import { MARKET_ITEMS, type PrivateState, type ViewModel } from '../../../default/src/types';
import type { GameState } from '../engine/types';

export interface PriceHistoryTracker {
  record(state: GameState): Record<string, number[]>;
  reset(): void;
}

export function createPriceHistory(turnsPerDay: number): PriceHistoryTracker {
  const series: Record<string, number[]> = {};
  let starts: Record<string, number> | null = null;

  return {
    record(state: GameState): Record<string, number[]> {
      const prices = state.market.prices ?? {};
      if (!starts) {
        starts = {};
        for (const { key } of MARKET_ITEMS) starts[key] = Number(prices[key as keyof typeof prices] ?? 0);
      }
      for (const { key } of MARKET_ITEMS) {
        const v = Number(prices[key as keyof typeof prices] ?? starts[key]);
        const buf = series[key] ?? (series[key] = []);
        buf.push(v);
        while (buf.length > turnsPerDay) buf.shift();
      }
      const out: Record<string, number[]> = {};
      for (const { key } of MARKET_ITEMS) {
        const buf = series[key] ?? [];
        const start = starts[key] ?? 0;
        const padCount = Math.max(0, turnsPerDay - buf.length);
        out[key] = new Array<number>(padCount).fill(start).concat(buf);
      }
      return out;
    },
    reset(): void {
      for (const k of Object.keys(series)) delete series[k];
      starts = null;
    },
  };
}

export function stateToView(state: GameState, priceHistory: Record<string, number[]>): ViewModel {
  const privates: (PrivateState | undefined)[] = state.privates.map((p) => p as unknown as PrivateState);
  return {
    day: state.day,
    hour: state.hour,
    farms: state.farms as unknown as ViewModel['farms'],
    market: state.market as unknown as ViewModel['market'],
    town: state.town as unknown as ViewModel['town'],
    privates,
    priceHistory,
  };
}
