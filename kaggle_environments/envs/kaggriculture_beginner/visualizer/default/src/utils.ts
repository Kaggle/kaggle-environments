import { CROP_FIRST_YIELD_DAY, CROP_SPRITE, PNG_SPRITES, READY_SPRITE_TYPES, type Crop } from './types';

export function spriteSrc(name: string): string {
  const png = PNG_SPRITES[name];
  if (png) return `/sprites/${png}.png`;
  return `/sprites/legacy/${name}.svg`;
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
