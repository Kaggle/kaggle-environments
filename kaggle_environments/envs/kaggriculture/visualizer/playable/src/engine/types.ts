/**
 * TypeScript types for kaggriculture game state. Mirrors the dict-shaped
 * structures used by `kaggriculture.py` so the porting maps 1:1.
 */

export type CropId = 'WHEAT' | 'CARROT' | 'TOMATO' | 'STRAWBERRY' | 'MELON';
export type AnimalId = 'GOOSE' | 'COW' | 'SHEEP';
export type AnimalProductId = 'EGG' | 'MILK' | 'WOOL';
export type StructureKind = 'COOP' | 'PASTURE';

export type CropProductId = CropId;
export type ProductId = CropProductId | AnimalProductId | 'FERTILIZER';

// Things that can sit in the shed (PRODUCTS + animals).
export type ShedItemId = ProductId | AnimalId;

// Anything one of the BUY_* market sub-ops can produce.
export type BuyableItemId = ProductId | CropId | AnimalId;

export type ShopId = ProductId | CropId; // shop products are restricted to non-fertilizer products

export type ShopName =
  | 'BAKERY'
  | 'PIZZA_SHOP'
  | 'BRUNCH_SPOT'
  | 'YARN_STORE'
  | 'ICE_CREAM_SHOP'
  | 'PET_CAFE'
  | 'SMOOTHIE_SHOP'
  | 'FARMERS_MARKET';

export type Quadrant = 'NW' | 'NE' | 'SW' | 'SE';

export interface CropSpec {
  seed: number;
  first_yield_day: number;
  max_yield_day: number;
  interval: number;
  max_yield: number;
  ongoing: boolean;
}

export interface AnimalSpec {
  cost: number;
  structure: StructureKind;
  first_yield_day: number;
  interval: number;
  max_held: number;
  product: AnimalProductId;
}

export type ShapeFunc = 'linear' | 'sq' | 'sqrt' | 'log' | 'log10';

export interface MarketParam {
  base: number;
  I0: number;
  T: number;
  below_func: ShapeFunc;
  below_target: number;
  above_func: ShapeFunc;
  above_target: number;
}

/** A locked-quadrant tile; sentinel string in the Python state too. */
export const LOCKED = 'LOCKED' as const;
export type LockedTile = typeof LOCKED;

export interface PlantTile {
  kind: 'PLANT';
  crop: CropId;
  planted_day: number;
  watered_today: boolean;
  consecutive_unwatered: number;
  yield_units: number;
  /** Step on which the plant becomes "overripe" and begins to decay. -1 for ongoing crops. */
  max_lifespan_step: number;
  /** Latest day on which the fertilizer bonus is still active. -1 if never fertilized. */
  fertilized_until_day: number;
}

export interface WeedTile {
  kind: 'WEED';
}

export interface EmptyStructureTile {
  kind: StructureKind;
  // No animal placed yet.
}

export interface AnimalTile {
  kind: StructureKind;
  animal: AnimalId;
  placed_day: number;
  yield_units: number;
  consecutive_unfed: number;
  fed_today: boolean;
  cared_today: boolean;
  fertilizer_available: boolean;
  pending_care_bonus: number;
}

/** All possible tile values. `null` = empty unlocked. */
export type Tile = null | LockedTile | PlantTile | WeedTile | EmptyStructureTile | AnimalTile;

export type Position = [number, number]; // [x, y]

export interface Farm {
  money: number;
  tiles: Tile[][]; // tiles[y][x]
  farmer: Position;
  hands: Position[];
  unlocked_quadrants: Quadrant[];
  hires_today: number;
}

export interface Private {
  shed: Partial<Record<ShedItemId, number>>;
  seeds: Record<CropId, number>;
  /** inventories[0] = main farmer, inventories[1..] = hand i-1. */
  inventories: Partial<Record<ShedItemId, number>>[];
}

export interface Market {
  inventory: Record<ProductId, number>;
  prices: Record<ProductId, number>;
  /** Only present if the env was configured with `marketParams` overrides. */
  params?: Record<ProductId, MarketParam>;
}

export interface Town {
  unlocked_shops: ShopName[];
}

/** Per-unit action: a single op for the farmer or one hand. */
export type UnitAction =
  | ['NORTH' | 'SOUTH' | 'EAST' | 'WEST' | 'PASS']
  | ['PICKUP', ShedItemId]
  | ['PICKUP', ShedItemId, number]
  | ['PLACE', ShedItemId]
  | ['PLACE', ShedItemId, number]
  | ['PLANT', CropId]
  | ['WATER']
  | ['HARVEST']
  | ['FERTILIZE']
  | ['DIG']
  | ['BUILD_COOP']
  | ['BUILD_PASTURE']
  | ['FEED']
  | ['COLLECT_FERTILIZER']
  | ['CARE'];

export type MarketOrder =
  | ['HIRE']
  | ['BUY_LAND']
  | ['BUY_SEED', CropId, number]
  | ['BUY_PRODUCT', 'WHEAT' | 'FERTILIZER', number]
  | ['BUY_ANIMAL', AnimalId, number]
  | ['SELL', ProductId, number];

export interface PlayerAction {
  farmer: UnitAction;
  hands: UnitAction[];
  market: MarketOrder[];
}

/**
 * Snapshot of all game state. In the Python env each agent's state has its
 * own observation; here we keep them flattened in one shape — observations
 * for each player are derived by filling in `player` and `private`.
 */
export interface GameState {
  step: number;
  day: number;
  hour: number;
  numAgents: number;
  /** Resolved input seed (used to drive end-of-day RNGs). */
  seed: number;
  farms: Farm[];
  privates: Private[];
  market: Market;
  town: Town;
  done: boolean;
  /** Final scores (money) — populated when `done` is true. */
  scores: number[];
}

export interface Config {
  episodeSteps: number;
  boardSize: number;
  startingMoney: number;
  maxMarketOrdersPerTurn: number;
  turnsPerDay: number;
  shedCapacity: number;
  weedSpawnChance: number;
  townShopUnlockInterval: number;
  townShopSellInterval: number;
  townCenterSellInterval: number;
  farmHandCostMult: number;
  /** Optional sparse overrides; merged onto MARKET_PARAMS at init time. */
  marketParams?: Partial<Record<ProductId, Partial<MarketParam>>>;
  /** Optional input seed. If undefined, the init helper draws a random one. */
  seed?: number;
}
