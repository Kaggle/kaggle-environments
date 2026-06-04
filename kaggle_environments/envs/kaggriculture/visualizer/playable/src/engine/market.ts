/**
 * Market pricing math: `_shape` + `market_price` ported from kaggriculture.py.
 * The pricing curves are pure functions of (item, inventory) — no rounding
 * surprises across JS/Python because both languages use IEEE-754 doubles and
 * the math here uses only +, -, *, /, sqrt, log, log10, and Math.pow(2,2) is
 * just multiplication.
 */

import { MARKET_PARAMS, PRICE_FLOOR, PRODUCTS } from './constants';
import type { Market, MarketParam, ProductId, ShapeFunc } from './types';

export function shape(func: ShapeFunc, xIn: number): number {
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
  }
}

export function marketPrice(item: ProductId, inventory: number, params?: Record<ProductId, MarketParam>): number {
  const p = (params ?? MARKET_PARAMS)[item];
  const { base, I0, T } = p;
  let price: number;
  if (inventory < I0) {
    const f = p.below_func;
    const amp = (p.below_target * base) / shape(f, T);
    price = base + amp * shape(f, I0 - inventory);
  } else {
    const f = p.above_func;
    const amp = (p.above_target * base) / shape(f, T);
    price = base - amp * shape(f, inventory - I0);
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
