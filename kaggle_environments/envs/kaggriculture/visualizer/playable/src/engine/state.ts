/**
 * Initial state builder + tile/farm/private helpers.
 * Mirrors `_initialize`, `_new_farm`, `_new_private`, `_new_market`,
 * `_new_town`, plus the supporting tile helpers in kaggriculture.py.
 */

import { ANIMALS, CROPS, DEFAULT_CONFIG, MARKET_PARAMS, PRODUCTS } from './constants';
import { resolveMarketParams } from './market';
import type {
  AnimalId,
  Config,
  CropId,
  Farm,
  GameState,
  Market,
  MarketParam,
  PlantTile,
  Position,
  Private,
  ProductId,
  Quadrant,
  Tile,
  Town,
} from './types';
import { LOCKED } from './types';

export function quadrantOf(x: number, y: number, boardSize: number): Quadrant {
  const half = Math.floor(boardSize / 2);
  const ns: 'N' | 'S' = y < half ? 'N' : 'S';
  const ew: 'W' | 'E' = x < half ? 'W' : 'E';
  return (ns + ew) as Quadrant;
}

/** Four inner-corner tiles around the central shed, in NWSE order. */
export function shedAccessTiles(boardSize: number): Position[] {
  const half = Math.floor(boardSize / 2);
  return [
    [half - 1, half - 1],
    [half, half - 1],
    [half - 1, half],
    [half, half],
  ];
}

export function isShedAdjacent(pos: Position, boardSize: number): boolean {
  const [x, y] = pos;
  for (const [sx, sy] of shedAccessTiles(boardSize)) {
    if (sx === x && sy === y) return true;
  }
  return false;
}

/** First free shed-access tile inside the (unlocked) NW quadrant. */
export function defaultSpawn(boardSize: number): Position {
  for (const tile of shedAccessTiles(boardSize)) {
    if (quadrantOf(tile[0], tile[1], boardSize) === 'NW') return [tile[0], tile[1]];
  }
  return [0, 0];
}

function initialTile(x: number, y: number, boardSize: number): Tile {
  return quadrantOf(x, y, boardSize) === 'NW' ? null : LOCKED;
}

export function newFarm(boardSize: number, startingMoney: number): Farm {
  const tiles: Tile[][] = [];
  for (let y = 0; y < boardSize; y++) {
    const row: Tile[] = [];
    for (let x = 0; x < boardSize; x++) row.push(initialTile(x, y, boardSize));
    tiles.push(row);
  }
  return {
    money: startingMoney,
    tiles,
    farmer: defaultSpawn(boardSize),
    hands: [],
    unlocked_quadrants: ['NW'],
    hires_today: 0,
  };
}

export function newPrivate(): Private {
  const shed: Partial<Record<string, number>> = {};
  for (const item of PRODUCTS) shed[item] = 0;
  for (const animal of Object.keys(ANIMALS) as AnimalId[]) shed[animal] = 0;
  const seeds = {} as Record<CropId, number>;
  for (const crop of Object.keys(CROPS) as CropId[]) seeds[crop] = 0;
  return {
    shed: shed as Private['shed'],
    seeds,
    inventories: [{}],
  };
}

export function newMarket(params?: Record<ProductId, MarketParam>): Market {
  const inv = {} as Record<ProductId, number>;
  const prices = {} as Record<ProductId, number>;
  const effective = params ?? MARKET_PARAMS;
  for (const item of PRODUCTS) {
    inv[item] = effective[item].I0;
    prices[item] = effective[item].base;
  }
  const market: Market = { inventory: inv, prices };
  if (params && params !== MARKET_PARAMS) market.params = params;
  return market;
}

export function newTown(): Town {
  return { unlocked_shops: [] };
}

/** Build a brand-new plant tile. Mirrors `_new_plant`. */
export function newPlant(crop: CropId, day: number, turnsPerDay: number): PlantTile {
  const cd = CROPS[crop];
  return {
    kind: 'PLANT',
    crop,
    planted_day: day,
    watered_today: false,
    consecutive_unwatered: 1, // planting day counts as unwatered
    yield_units: cd.ongoing ? 0 : 1,
    max_lifespan_step: cd.ongoing ? -1 : (day + cd.max_yield_day + 1) * turnsPerDay,
    fertilized_until_day: -1,
  };
}

/** Build a freshly-placed animal tile. Mirrors `_new_animal`. */
export function newAnimal(animal: AnimalId, day: number) {
  const a = ANIMALS[animal];
  return {
    kind: a.structure,
    animal,
    placed_day: day,
    yield_units: 0,
    consecutive_unfed: 0,
    fed_today: false,
    cared_today: false,
    fertilizer_available: false,
    pending_care_bonus: 0,
  };
}

/** Apply an optional partial config on top of the defaults. */
export function resolveConfig(overrides?: Partial<Config>): Config {
  return { ...DEFAULT_CONFIG, ...(overrides ?? {}) };
}

/**
 * Build the initial GameState for a match. Mirrors `_initialize` — except
 * the resolved seed is required (we don't draw from `random` here; the
 * caller passes one in or the worker generates it).
 */
export function initGameState(numAgents: number, config: Config, seed: number): GameState {
  const farms: Farm[] = [];
  const privates: Private[] = [];
  for (let i = 0; i < numAgents; i++) {
    farms.push(newFarm(config.boardSize, config.startingMoney));
    privates.push(newPrivate());
  }
  const params = config.marketParams ? resolveMarketParams(config.marketParams) : undefined;
  return {
    step: 0,
    day: 0,
    hour: 0,
    numAgents,
    seed,
    farms,
    privates,
    market: newMarket(params),
    town: newTown(),
    done: false,
    scores: new Array(numAgents).fill(0),
  };
}

/** Pick a fresh seed when the caller doesn't supply one. Range matches Python's `random.randrange(2**31)`. */
export function pickSeed(): number {
  // Use crypto where available so two tabs don't accidentally pick the same seed.
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const arr = new Uint32Array(1);
    crypto.getRandomValues(arr);
    return arr[0] & 0x7fffffff;
  }
  return Math.floor(Math.random() * 0x80000000);
}

/** Spawn position for a newly hired hand: first free shed-access tile (NWSE order); ties by min occupancy. */
export function spawnHand(farm: Farm, boardSize: number): Position {
  const tiles = shedAccessTiles(boardSize);
  const occupants: number[] = tiles.map(() => 0);
  const allPos: Position[] = [farm.farmer, ...farm.hands];
  for (const pos of allPos) {
    const idx = tiles.findIndex(([tx, ty]) => tx === pos[0] && ty === pos[1]);
    if (idx >= 0) occupants[idx]++;
  }
  let best = 0;
  for (let i = 1; i < tiles.length; i++) {
    if (occupants[i] < occupants[best]) best = i;
    // ties: keep earlier (NWSE order)
  }
  return [tiles[best][0], tiles[best][1]];
}
