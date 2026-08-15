/**
 * Market pricing math: `_shape` + `market_price` ported from kaggriculture.py.
 * The pricing curves are pure functions of (item, inventory), and both
 * languages use IEEE-754 doubles for the +, -, *, /, sqrt, log, log10 here.
 *
 * Caveat: the final rounding can differ by $1 when the unrounded price lands
 * exactly on .5, because Python's round() is banker's rounding while
 * Math.round() is half-up. With the default MARKET_PARAMS this happens at a
 * handful of carrot inventory levels (e.g. I0+450).
 */

import { MARKET_PARAMS, PRICE_FLOOR, PRODUCTS } from './constants';
import type { Market, MarketParam, ProductId, ShapeFunc } from './types';

/** Matches HINGE_GAIN in kaggriculture.py. */
const HINGE_GAIN = 8;

export function shape(func: ShapeFunc, xIn: number, T?: number): number {
  const x = Math.max(0, xIn);
  switch (func) {
    case 'linear':
      return x;
    case 'sq':
      return x * x;
    case 'sqrt':
      return Math.sqrt(x);
    case 'log':
      return Math.log(1 + x);
    case 'log10':
      return Math.log10(1 + x);
    case 'hinge': {
      // Degenerates to linear if T is missing or non-positive.
      if (!T || T <= 0) return x;
      const u = x / T;
      const over = Math.max(0, u - 1);
      return u + HINGE_GAIN * over * over;
    }
  }
}

export function marketPrice(item: ProductId, inventory: number, params?: Record<ProductId, MarketParam>): number {
  const p = (params ?? MARKET_PARAMS)[item];
  const { base, I0, T } = p;
  let price: number;
  if (inventory < I0) {
    const f = p.below_func;
    const amp = (p.below_target * base) / shape(f, T, T);
    price = base + amp * shape(f, I0 - inventory, T);
  } else {
    const f = p.above_func;
    const amp = (p.above_target * base) / shape(f, T, T);
    price = base - amp * shape(f, inventory - I0, T);
  }
  return Math.max(PRICE_FLOOR, Math.round(price));
}

export function refreshPrices(market: Market): void {
  for (const item of PRODUCTS) {
    market.prices[item] = marketPrice(item, market.inventory[item], market.params);
  }
}

/** Merge sparse per-product overrides onto the defaults; same shape as `_resolve_market_params`. */
export function resolveMarketParams(
  overrides?: Partial<Record<ProductId, Partial<MarketParam>>>
): Record<ProductId, MarketParam> {
  const out = {} as Record<ProductId, MarketParam>;
  for (const item of PRODUCTS) {
    out[item] = { ...MARKET_PARAMS[item] };
  }
  if (!overrides) return out;
  for (const item of PRODUCTS) {
    const patch = overrides[item];
    if (patch) out[item] = { ...out[item], ...patch };
  }
  return out;
}
