export const CROPS = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON'] as const;
export type Crop = (typeof CROPS)[number];

export const CROP_FIRST_YIELD_DAY: Record<Crop, number> = {
  WHEAT: 2,
  CARROT: 2,
  TOMATO: 8,
  STRAWBERRY: 10,
  MELON: 10,
};

export const CROP_SPRITE: Record<Crop, string> = {
  WHEAT: 'wheat',
  CARROT: 'carrot',
  TOMATO: 'tomato',
  STRAWBERRY: 'strawberry',
  MELON: 'melon',
};

export const READY_SPRITE_TYPES = new Set<Crop>(['CARROT', 'TOMATO', 'STRAWBERRY', 'MELON']);

export const PNG_SPRITES: Record<string, string> = {
  wheat: 'wheat_full',
  carrot: 'carrot_full',
  tomato: 'tomato_full',
  strawberry: 'strawberry_full',
  melon: 'melon_full',
  coin: 'coin',
  farmer_p1: 'farmer_p1',
  farmer_p2: 'farmer_p2',
  soil_dry: 'soil_dry',
  soil_watered: 'soil_watered',
  sprout: 'sprout',
  midgrowth: 'midgrowth',
  carrot_ready: 'carrot_ready',
  tomato_ready: 'tomato_ready',
  strawberry_ready: 'strawberry_ready',
  melon_ready: 'melon_ready',
};

export interface BoardSize {
  rows: number;
  cols: number;
}

export interface Tile {
  crop: Crop;
  planted_day: number;
  watered_today: boolean;
  yield_units: number;
  consecutive_unwatered: number;
  max_lifespan_step: number;
}

export interface Farm {
  money: number;
  seeds: Record<Crop, number>;
  farmer: [number, number];
  tiles: Array<Array<Tile | null>>;
}

export interface Observation {
  step?: number;
  day?: number;
  hour?: number;
  farms?: Farm[];
}

export interface CellRefs {
  el: HTMLElement;
  baseImg: HTMLImageElement;
  baseSprite: string;
  objectSlot: HTMLElement;
  objectImg: HTMLImageElement;
  objectSprite: string;
  agentSlot: HTMLElement;
  agentImg: HTMLImageElement;
  agentSprite: string;
}

export interface SeedSlotRefs {
  count: HTMLElement;
  lastCount: number;
}

export interface PlayerRefs {
  panel: HTMLElement;
  balance: HTMLElement;
  lastBalance: number;
  cells: CellRefs[][];
  seeds: Record<Crop, SeedSlotRefs>;
  farmerCell: CellRefs | null;
}

export interface LayoutRefs {
  dayValue: HTMLElement;
  lastDay: number;
  turnValue: HTMLElement;
  lastTurn: number;
  players: PlayerRefs[];
}
