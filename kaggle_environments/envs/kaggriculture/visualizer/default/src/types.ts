export const SEGMENT = 5;
export const INVENTORY_SLOTS = 20;

export const TOWN_GRID_COLS = 5;
export const TOWN_GRID_ROWS = 5;
export const TOWN_CENTER_INDEX = Math.floor((TOWN_GRID_COLS * TOWN_GRID_ROWS) / 2);

export const SURROUNDING_BUILDINGS: Record<number, string> = {
  6: 'bakery',
  7: 'pizza',
  8: 'brunch',
  11: 'yarn',
  13: 'icecream',
  16: 'petcafe',
  17: 'smoothie',
  18: 'farmersmarket',
};

export const MARKET_ITEMS: { sprite: string; key: string }[] = [
  { sprite: 'wheat', key: 'wheat' },
  { sprite: 'carrot', key: 'carrot' },
  { sprite: 'tomato', key: 'tomato' },
  { sprite: 'strawberry', key: 'strawberry' },
  { sprite: 'melon', key: 'melon' },
  { sprite: 'egg', key: 'egg' },
  { sprite: 'milk', key: 'milk' },
  { sprite: 'wool', key: 'wool' },
];

// Plant types where the "ready" sprite should swap to a dedicated `_ready` PNG.
export const READY_SPRITE_TYPES = new Set(['carrot', 'tomato', 'strawberry', 'melon']);

export interface BoardSize {
  rows: number;
  cols: number;
}

export interface PlantCell {
  row: number;
  col: number;
  type: 'plant';
  plant: string;
  stage: string;
  wateredToday?: boolean;
  fertilizedDaysLeft?: number;
}

export interface StructureCell {
  row: number;
  col: number;
  type: 'coop' | 'pasture';
  animal: { kind: string; fedToday?: boolean } | null;
}

export interface WeedCell {
  row: number;
  col: number;
  type: 'weed';
}

export type Cell = PlantCell | StructureCell | WeedCell;

export interface AgentEntity {
  role: 'farmer' | 'farmhand';
  variant?: number;
  row: number;
  col: number;
  inventory?: Record<string, number>;
}

export interface PlayerState {
  coins: number;
  shed?: Record<string, number>;
  farm: { size: [number, number]; unlockedSegments: number[]; cells: Cell[] };
  agents: AgentEntity[];
}

export interface MarketEntry {
  current: number;
  base: number;
}

export interface Observation {
  step?: number;
  day?: number;
  turnOfDay?: number;
  market?: Record<string, MarketEntry>;
  townBuildings?: { active: string[] };
  players?: PlayerState[];
}

export interface CellRefs {
  el: HTMLElement;
  segment: number;
  baseImg: HTMLImageElement;
  objectSlot: HTMLElement;
  agentSlot: HTMLElement;
}

export interface InventorySlotRefs {
  icon: HTMLElement;
  count: HTMLElement;
}

export interface PlayerRefs {
  panel: HTMLElement;
  balance: HTMLElement;
  cells: CellRefs[][]; // [row][col]
  inventory: InventorySlotRefs[];
}

export interface LayoutRefs {
  dayValue: HTMLElement;
  turnValue: HTMLElement;
  marketItems: Record<string, { item: HTMLElement; price: HTMLElement }>;
  shopSlots: HTMLElement[];
  players: PlayerRefs[];
}
