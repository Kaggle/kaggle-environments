import { getStepData } from '@kaggle-environments/core';
import { MARKET_ITEMS, READY_SPRITE_TYPES, type PrivateState, type ViewModel } from './types';

import bakery from './assets/sprites/bakery.png';
import brunch_spot from './assets/sprites/brunch_spot.png';
import bush_border_horizontal from './assets/sprites/bush_border_horizontal.png';
import carrot_full from './assets/sprites/carrot_full.png';
import carrot_ready from './assets/sprites/carrot_ready.png';
import cobblestone_cell from './assets/sprites/cobblestone_cell.png';
import coin from './assets/sprites/coin.png';
import coop from './assets/sprites/coop.png';
import cow from './assets/sprites/cow.png';
import egg from './assets/sprites/egg.png';
import farmer_p1 from './assets/sprites/farmer_p1.png';
import farmer_p2 from './assets/sprites/farmer_p2.png';
import farmers_market from './assets/sprites/farmers_market.png';
import farmhand_1 from './assets/sprites/farmhand_1.png';
import farmhand_2 from './assets/sprites/farmhand_2.png';
import farmhand_3 from './assets/sprites/farmhand_3.png';
import fence_horizontal from './assets/sprites/fence_horizontal.png';
import fence_vertical from './assets/sprites/fence_vertical.png';
import fertilizer from './assets/sprites/fertilizer.png';
import flowers_horizontal from './assets/sprites/flowers_horizontal.png';
import flowers_vertical from './assets/sprites/flowers_vertical.png';
import goose from './assets/sprites/goose.png';
import grass_bg from './assets/sprites/grass_bg.svg';
import ice_cream_shop from './assets/sprites/ice_cream_shop.png';
import locked_cell from './assets/sprites/locked_cell.png';
import melon_full from './assets/sprites/melon_full.png';
import melon_ready from './assets/sprites/melon_ready.png';
import midgrowth from './assets/sprites/midgrowth.png';
import milk from './assets/sprites/milk.png';
import pasture from './assets/sprites/pasture.png';
import pet_cafe from './assets/sprites/pet_cafe.png';
import pizza_shop from './assets/sprites/pizza_shop.png';
import seed_packet from './assets/sprites/seed_packet.png';
import shed from './assets/sprites/shed.png';
import sheep from './assets/sprites/sheep.png';
import smoothie_shop from './assets/sprites/smoothie_shop.png';
import soil_dry from './assets/sprites/soil_dry.png';
import soil_watered from './assets/sprites/soil_watered.png';
import sprout from './assets/sprites/sprout.png';
import strawberry_full from './assets/sprites/strawberry_full.png';
import strawberry_ready from './assets/sprites/strawberry_ready.png';
import tomato_full from './assets/sprites/tomato_full.png';
import tomato_ready from './assets/sprites/tomato_ready.png';
import town_center from './assets/sprites/town_center.png';
import town_sign from './assets/sprites/town_sign.png';
import weed from './assets/sprites/weed.png';
import wheat_full from './assets/sprites/wheat_full.png';
import wood_bg from './assets/sprites/wood_bg.svg';
import wool from './assets/sprites/wool.png';
import yarn_store from './assets/sprites/yarn_store.png';

import market_carrot from './assets/sprites/market_carrot.png';
import market_melon from './assets/sprites/market_melon.png';
import market_strawberry from './assets/sprites/market_strawberry.png';
import market_tomato from './assets/sprites/market_tomato.png';
import market_wheat from './assets/sprites/market_wheat.png';
import market_egg from './assets/sprites/market_egg.png';
import market_milk from './assets/sprites/market_milk.png';
import market_wool from './assets/sprites/market_wool.png';

// Map logical sprite names used in the renderer to imported asset URLs.
// Keys can be the logical name (e.g. 'pizza') or the file basename
// (e.g. 'pizza_shop') — both forms are looked up below.
const SPRITE_URLS: Record<string, string> = {
  bakery,
  brunch_spot,
  brunch: brunch_spot,
  bush_border_horizontal,
  carrot_full,
  carrot: carrot_full,
  carrot_ready,
  cobblestone_cell,
  coin,
  coop,
  cow,
  egg,
  farmer_p1,
  farmer_p2,
  farmers_market,
  farmersmarket: farmers_market,
  farmhand_1,
  farmhand_2,
  farmhand_3,
  fence_horizontal,
  fence_vertical,
  fertilizer,
  flowers_horizontal,
  flowers_vertical,
  goose,
  ice_cream_shop,
  icecream: ice_cream_shop,
  locked_cell,
  melon_full,
  melon: melon_full,
  melon_ready,
  midgrowth,
  milk,
  pasture,
  pet_cafe,
  petcafe: pet_cafe,
  pizza_shop,
  pizza: pizza_shop,
  seed_packet,
  shed,
  sheep,
  smoothie_shop,
  smoothie: smoothie_shop,
  soil_dry,
  soil_watered,
  sprout,
  strawberry_full,
  strawberry: strawberry_full,
  strawberry_ready,
  tomato_full,
  tomato: tomato_full,
  tomato_ready,
  town_center,
  town_sign,
  weed,
  wheat_full,
  wheat: wheat_full,
  wool,
  yarn_store,
  yarn: yarn_store,

  // New illustrative market sprites
  market_carrot,
  market_melon,
  market_strawberry,
  market_tomato,
  market_wheat,
  market_egg,
  market_milk,
  market_wool,
};

export const BG_URLS = {
  grass: grass_bg,
  wood: wood_bg,
  cobble: cobblestone_cell,
};

export function spriteSrc(name: string): string {
  return SPRITE_URLS[name] ?? '';
}

export function marketSpriteSrc(name: string): string {
  const marketName = `market_${name}`;
  return SPRITE_URLS[marketName] ?? SPRITE_URLS[name] ?? '';
}

export function plantSprite(plant: string, stage: string): string {
  if (stage === 'sprout') return 'sprout';
  if (stage === 'midgrowth') return 'midgrowth';
  if (stage === 'ready' && READY_SPRITE_TYPES.has(plant)) {
    return `${plant}_ready`;
  }
  return plant;
}

export function clearChildren(el: Element): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function setCellSprite(slot: Element, src: string, alt: string, extraClass = '', title?: string): void {
  const cls = `cell-sprite ${extraClass}`.trim();
  const titleAttr = title ? ` title="${title}"` : '';
  slot.innerHTML = `<img class="${cls}" src="${src}" alt="${alt}"${titleAttr} />`;
}

export function titleCase(s: string): string {
  return s
    .toLowerCase()
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export function pricesAt(replay: any, step: number): Record<string, number> {
  const data = getStepData(replay, step) as any;
  if (!data) return {};
  const entry = Array.isArray(data) ? data[0] : data;
  return entry?.observation?.market?.prices ?? {};
}

export function buildPriceHistory(replay: any, step: number, windowSize: number): Record<string, number[]> {
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

export function buildView(replay: any, step: number, turnsPerDay: number): ViewModel | null {
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
