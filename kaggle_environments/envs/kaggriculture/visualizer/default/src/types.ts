export const SEGMENT = 5;
export const INVENTORY_SLOTS = 20;

// 3x4 grid: top row holds the town center (left), town sign (middle), and the
// first shop (right). The middle column below the sign stays empty except for
// the bottom slot, which holds the bakery to terminate the path. The remaining
// 7 shops line the left and right columns. Shops unlock one per 3 in-game days.
export const TOWN_GRID_COLS = 3;
export const TOWN_GRID_ROWS = 4;
export const TOWN_CENTER_INDEX = 0;
export const TOWN_SIGN_INDEX = 1;

// segId = segR*2 + segC, where segR/segC come from row/col / SEGMENT.
// (0,0)=NW, (0,1)=NE, (1,0)=SW, (1,1)=SE.
export const QUADRANT_BY_SEGMENT: Record<number, string> = {
  0: 'NW',
  1: 'NE',
  2: 'SW',
  3: 'SE',
};

// Shop slot index in the 3x4 grid (skipping center=0, sign=1, and the empty
// inner cells 4 and 7) -> { interpreter shop key, sprite name, label }.
export const SURROUNDING_BUILDINGS: Record<number, { shop: string; sprite: string; label: string }> = {
  2: { shop: 'PIZZA_SHOP', sprite: 'pizza', label: 'Pizza Shop' },
  3: { shop: 'BRUNCH_SPOT', sprite: 'brunch', label: 'Brunch Spot' },
  5: { shop: 'YARN_STORE', sprite: 'yarn', label: 'Yarn Store' },
  6: { shop: 'ICE_CREAM_SHOP', sprite: 'icecream', label: 'Ice Cream Shop' },
  8: { shop: 'PET_CAFE', sprite: 'petcafe', label: 'Pet Cafe' },
  9: { shop: 'SMOOTHIE_SHOP', sprite: 'smoothie', label: 'Smoothie Shop' },
  10: { shop: 'BAKERY', sprite: 'bakery', label: 'Bakery' },
  11: { shop: 'FARMERS_MARKET', sprite: 'farmersmarket', label: "Farmers' Market" },
};

// Visible market items. `key` is the interpreter's PRODUCTS key; `sprite` is the asset name.
export const MARKET_ITEMS: { sprite: string; key: string }[] = [
  { sprite: 'wheat', key: 'WHEAT' },
  { sprite: 'carrot', key: 'CARROT' },
  { sprite: 'tomato', key: 'TOMATO' },
  { sprite: 'strawberry', key: 'STRAWBERRY' },
  { sprite: 'melon', key: 'MELON' },
  { sprite: 'egg', key: 'EGG' },
  { sprite: 'milk', key: 'MILK' },
  { sprite: 'wool', key: 'WOOL' },
];

// Plant types where the "ready" sprite should swap to a dedicated `_ready` PNG.
export const READY_SPRITE_TYPES = new Set(['carrot', 'tomato', 'strawberry', 'melon']);

// first_yield_day per crop, mirrored from CROPS in kaggriculture.py. Used to
// pick sprout / midgrowth / ready sprites since the replay only carries
// planted_day + yield_units.
export const CROP_FIRST_YIELD_DAY: Record<string, number> = {
  WHEAT: 2,
  CARROT: 2,
  TOMATO: 8,
  STRAWBERRY: 10,
  MELON: 10,
};

export interface BoardSize {
  rows: number;
  cols: number;
}

// Raw tile shapes as they appear in farm.tiles[y][x].
export type RawTile =
  | null
  | 'LOCKED'
  | {
      kind: 'PLANT';
      crop: string;
      planted_day: number;
      watered_today: boolean;
      yield_units: number;
      fertilized_until_day: number;
    }
  | { kind: 'WEED' }
  | { kind: 'COOP' | 'PASTURE'; animal?: string; fed_today?: boolean; cared_today?: boolean; yield_units?: number };

export interface FarmPublic {
  money: number;
  tiles: RawTile[][];
  farmer: [number, number]; // [x, y]
  hands: [number, number][]; // list of [x, y]
  unlocked_quadrants: string[]; // e.g. ['NW', 'NE']
  hires_today: number;
}

export interface MarketPublic {
  prices: Record<string, number>;
  inventory: Record<string, number>;
}

export interface TownPublic {
  unlocked_shops: string[];
}

export interface PrivateState {
  shed: Record<string, number>;
  seeds: Record<string, number>;
  inventories: Record<string, number>[];
}

// Combined view assembled by the renderer from both agents' step entries.
export interface ViewModel {
  day: number;
  hour: number;
  farms: FarmPublic[];
  market: MarketPublic;
  town: TownPublic;
  privates: (PrivateState | undefined)[];
  // Per-item price series for the most recent `turnsPerDay` steps; padded on
  // the left with the starting price when the game is younger than one day.
  priceHistory: Record<string, number[]>;
}

export interface CellRefs {
  el: HTMLElement;
  segment: number;
  baseImg: HTMLImageElement;
  objectSlot: HTMLElement;
  agentSlot: HTMLElement;
  // Cached "what we last wrote" keys; if unchanged we skip the DOM write so
  // the browser doesn't tear down + re-decode the <img> every step (causing
  // a visible flash).
  lastBaseKey?: string;
  lastObjectKey?: string;
  lastAgentKey?: string;
}

export interface InventorySlotRefs {
  icon: HTMLElement;
  count: HTMLElement;
  lastIconKey?: string;
  lastCount?: string;
}

export interface PlayerRefs {
  panel: HTMLElement;
  balance: HTMLElement;
  cells: CellRefs[][]; // [row][col]
  inventory: InventorySlotRefs[];
}

export interface DialogRefs {
  overlay: HTMLElement;
  title: HTMLElement;
  body: HTMLElement;
  closeBtn: HTMLElement;
}

export interface LayoutRefs {
  dayValues: HTMLElement[];
  turnValues: HTMLElement[];
  marketItems: Record<
    string,
    { item: HTMLElement; price: HTMLElement; sparkPath: SVGPathElement; lastSparkKey?: string }
  >;
  shopSlots: HTMLElement[];
  players: PlayerRefs[];
  dialog: DialogRefs;
}
