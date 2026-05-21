/**
 * Static game data: crop & animal tables, market parameters, shops.
 * Mirrors the top-of-file constants in `kaggriculture.py`.
 */

import type { AnimalId, AnimalSpec, CropId, CropSpec, MarketParam, ProductId, ShopId, ShopName } from './types';

export const CROPS: Record<CropId, CropSpec> = {
  WHEAT: { seed: 10, first_yield_day: 2, max_yield_day: 4, interval: 0, max_yield: 6, ongoing: false },
  CARROT: { seed: 20, first_yield_day: 2, max_yield_day: 3, interval: 0, max_yield: 4, ongoing: false },
  TOMATO: { seed: 50, first_yield_day: 8, max_yield_day: 8, interval: 1, max_yield: 4, ongoing: true },
  STRAWBERRY: { seed: 100, first_yield_day: 10, max_yield_day: 10, interval: 2, max_yield: 4, ongoing: true },
  MELON: { seed: 80, first_yield_day: 10, max_yield_day: 12, interval: 0, max_yield: 6, ongoing: false },
};

export const ANIMALS: Record<AnimalId, AnimalSpec> = {
  GOOSE: { cost: 300, structure: 'COOP', first_yield_day: 4, interval: 1, max_held: 4, product: 'EGG' },
  COW: { cost: 600, structure: 'PASTURE', first_yield_day: 8, interval: 2, max_held: 6, product: 'MILK' },
  SHEEP: { cost: 500, structure: 'PASTURE', first_yield_day: 6, interval: 3, max_held: 6, product: 'WOOL' },
};

export const PRODUCTS: ProductId[] = [
  'WHEAT',
  'CARROT',
  'TOMATO',
  'STRAWBERRY',
  'MELON',
  'EGG',
  'MILK',
  'WOOL',
  'FERTILIZER',
];

export const MARKET_I0 = 10000;
export const PRICE_FLOOR = 1;

export const MARKET_PARAMS: Record<ProductId, MarketParam> = {
  WHEAT: {
    base: 25,
    I0: MARKET_I0,
    T: 400,
    below_func: 'sqrt',
    below_target: 0.8,
    above_func: 'log',
    above_target: 0.2,
  },
  CARROT: {
    base: 35,
    I0: MARKET_I0,
    T: 450,
    below_func: 'log',
    below_target: 0.2,
    above_func: 'sqrt',
    above_target: 0.7,
  },
  TOMATO: {
    base: 60,
    I0: MARKET_I0,
    T: 200,
    below_func: 'linear',
    below_target: 0.4,
    above_func: 'sqrt',
    above_target: 0.6,
  },
  STRAWBERRY: {
    base: 120,
    I0: MARKET_I0,
    T: 100,
    below_func: 'sqrt',
    below_target: 0.7,
    above_func: 'linear',
    above_target: 0.4,
  },
  MELON: {
    base: 250,
    I0: MARKET_I0,
    T: 300,
    below_func: 'log',
    below_target: 0.2,
    above_func: 'sq',
    above_target: 0.9,
  },
  EGG: {
    base: 50,
    I0: MARKET_I0,
    T: 332,
    below_func: 'linear',
    below_target: 0.4,
    above_func: 'log',
    above_target: 0.2,
  },
  MILK: {
    base: 160,
    I0: MARKET_I0,
    T: 122,
    below_func: 'sqrt',
    below_target: 0.6,
    above_func: 'linear',
    above_target: 0.4,
  },
  WOOL: { base: 200, I0: MARKET_I0, T: 105, below_func: 'log', below_target: 0.2, above_func: 'sq', above_target: 0.8 },
  FERTILIZER: {
    base: 100,
    I0: MARKET_I0,
    T: 200,
    below_func: 'linear',
    below_target: 0.4,
    above_func: 'linear',
    above_target: 0.4,
  },
};

// (dx, dy) — y grows downward, matching the Python convention.
export const FARMER_MOVES: Record<'NORTH' | 'SOUTH' | 'EAST' | 'WEST', [number, number]> = {
  NORTH: [0, -1],
  SOUTH: [0, 1],
  EAST: [1, 0],
  WEST: [-1, 0],
};

// NW is always unlocked at game start; players unlock the rest in this order.
export const LAND_ORDER = ['NE', 'SW', 'SE'] as const;
export const LAND_PRICES = [1000, 2000, 4000] as const;

// n-th hire of the day costs FARM_HAND_COST_MULT * fib(n).
// fib here is indexed so fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5...
export const FARM_HAND_COST_MULT = 10;

export const SHOPS: Record<ShopName, ShopId[]> = {
  BAKERY: ['EGG', 'WHEAT'],
  PIZZA_SHOP: ['MILK', 'TOMATO', 'WHEAT'],
  BRUNCH_SPOT: ['EGG', 'WHEAT', 'STRAWBERRY'],
  YARN_STORE: ['WOOL'],
  ICE_CREAM_SHOP: ['STRAWBERRY', 'MILK', 'WHEAT'],
  PET_CAFE: ['CARROT'],
  SMOOTHIE_SHOP: ['STRAWBERRY', 'MILK'],
  FARMERS_MARKET: ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY'],
};

export const SHOP_NAMES: ShopName[] = Object.keys(SHOPS) as ShopName[];

export const TOWN_CENTER_PRODUCTS: ProductId[] = PRODUCTS.filter((p) => p !== 'FERTILIZER');

// Highest threshold first — matches the next-match-wins lookup in Python.
export const TOWN_CENTER_DEMAND_SCHEDULE: ReadonlyArray<readonly [number, number]> = [
  [20, 4],
  [10, 2],
  [0, 1],
];

// Default configuration — mirrors kaggriculture.json's `configuration.*.default`.
export const DEFAULT_CONFIG = {
  episodeSteps: 720,
  boardSize: 10,
  startingMoney: 2000,
  maxMarketOrdersPerTurn: 10,
  turnsPerDay: 24,
  shedCapacity: 100,
  weedSpawnChance: 0.005,
  townShopUnlockInterval: 3,
  townShopSellInterval: 2,
  townCenterSellInterval: 6,
  farmHandCostMult: FARM_HAND_COST_MULT,
} as const;
