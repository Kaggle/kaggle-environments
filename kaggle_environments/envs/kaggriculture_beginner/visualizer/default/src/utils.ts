import { CROP_FIRST_YIELD_DAY, CROP_SPRITE, PNG_SPRITES, READY_SPRITE_TYPES, type Crop } from './types';

import carrot_full from './assets/sprites/carrot_full.png';
import carrot_ready from './assets/sprites/carrot_ready.png';
import coin from './assets/sprites/coin.png';
import farmer_p1 from './assets/sprites/farmer_p1.png';
import farmer_p2 from './assets/sprites/farmer_p2.png';
import grass_bg from './assets/sprites/grass_bg.svg';
import melon_full from './assets/sprites/melon_full.png';
import melon_ready from './assets/sprites/melon_ready.png';
import midgrowth from './assets/sprites/midgrowth.png';
import seed_carrot from './assets/sprites/seed_carrot.png';
import seed_melon from './assets/sprites/seed_melon.png';
import seed_strawberry from './assets/sprites/seed_strawberry.png';
import seed_tomato from './assets/sprites/seed_tomato.png';
import seed_wheat from './assets/sprites/seed_wheat.png';
import soil_dry from './assets/sprites/soil_dry.png';
import soil_watered from './assets/sprites/soil_watered.png';
import sprout from './assets/sprites/sprout.png';
import strawberry_full from './assets/sprites/strawberry_full.png';
import strawberry_ready from './assets/sprites/strawberry_ready.png';
import tomato_full from './assets/sprites/tomato_full.png';
import tomato_ready from './assets/sprites/tomato_ready.png';
import wheat_full from './assets/sprites/wheat_full.png';
import wood_bg from './assets/sprites/wood_bg.svg';

const SPRITE_URLS: Record<string, string> = {
  carrot_full,
  carrot_ready,
  coin,
  farmer_p1,
  farmer_p2,
  melon_full,
  melon_ready,
  midgrowth,
  seed_carrot,
  seed_melon,
  seed_strawberry,
  seed_tomato,
  seed_wheat,
  soil_dry,
  soil_watered,
  sprout,
  strawberry_full,
  strawberry_ready,
  tomato_full,
  tomato_ready,
  wheat_full,
};

export const BG_URLS = {
  grass: grass_bg,
  wood: wood_bg,
};

export function spriteSrc(name: string): string {
  const png = PNG_SPRITES[name];
  return SPRITE_URLS[png ?? name] ?? '';
}

export function plantSprite(crop: Crop, ageDays: number): string {
  const firstYield = CROP_FIRST_YIELD_DAY[crop];
  if (ageDays <= 0) return 'sprout';
  if (ageDays < firstYield) return 'midgrowth';
  const base = CROP_SPRITE[crop];
  if (READY_SPRITE_TYPES.has(crop)) return `${base}_ready`;
  return base;
}

export function makeHiddenImg(extraClass = ''): HTMLImageElement {
  const img = document.createElement('img');
  img.className = `cell-sprite ${extraClass}`.trim();
  img.alt = '';
  img.style.display = 'none';
  return img;
}
