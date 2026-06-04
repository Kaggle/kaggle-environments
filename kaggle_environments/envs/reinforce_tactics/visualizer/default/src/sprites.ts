// Sprite assets for the Reinforce Tactics visualizer.
//
// Placeholder PNGs live in ./assets/sprites/ and are imported here so
// Vite bundles them as data URLs (the build inlines everything into a
// single index.html via vite-plugin-singlefile). When the art team
// delivers final art, dropping replacement files at the same paths is
// all that's needed.

import warriorUrl from './assets/sprites/warrior.png';
import mageUrl from './assets/sprites/mage.png';
import clericUrl from './assets/sprites/cleric.png';
import archerUrl from './assets/sprites/archer.png';
import knightUrl from './assets/sprites/knight.png';
import rogueUrl from './assets/sprites/rogue.png';
import sorcererUrl from './assets/sprites/sorcerer.png';
import barbarianUrl from './assets/sprites/barbarian.png';
import hqUrl from './assets/sprites/headquarters.png';
import buildingUrl from './assets/sprites/building.png';
import towerUrl from './assets/sprites/tower.png';
import grassUrl from './assets/sprites/grass.png';
import forestUrl from './assets/sprites/forest.png';
import mountainUrl from './assets/sprites/mountain.png';
import waterUrl from './assets/sprites/water.png';
import oceanUrl from './assets/sprites/ocean.png';
import roadUrl from './assets/sprites/road.png';

// Engine codes -> sprite URL.
const UNIT_URLS: Record<string, string> = {
  W: warriorUrl,
  M: mageUrl,
  C: clericUrl,
  A: archerUrl,
  K: knightUrl,
  R: rogueUrl,
  S: sorcererUrl,
  B: barbarianUrl,
};

const STRUCT_URLS: Record<string, string> = {
  h: hqUrl,
  b: buildingUrl,
  t: towerUrl,
};

// Terrain codes match the engine's TileType values.
const TERRAIN_URLS: Record<string, string> = {
  p: grassUrl,
  f: forestUrl,
  m: mountainUrl,
  w: waterUrl,
  o: oceanUrl,
  r: roadUrl,
};

const cache: Record<string, HTMLImageElement> = {};

// Callback fired the first time any pending sprite finishes loading,
// so the renderer can request a redraw without polling.
let reloadCallback: (() => void) | null = null;

function preload(key: string, url: string): HTMLImageElement {
  if (cache[key]) return cache[key];
  const img = new Image();
  img.onload = () => {
    if (reloadCallback) reloadCallback();
  };
  img.src = url;
  cache[key] = img;
  return img;
}

// Kick off all loads immediately at module import.
for (const [k, u] of Object.entries(UNIT_URLS)) preload(`u:${k}`, u);
for (const [k, u] of Object.entries(STRUCT_URLS)) preload(`s:${k}`, u);
for (const [k, u] of Object.entries(TERRAIN_URLS)) preload(`t:${k}`, u);

export function getUnitSprite(type: string): HTMLImageElement | null {
  return cache[`u:${type}`] ?? null;
}

export function getStructureSprite(type: string): HTMLImageElement | null {
  return cache[`s:${type}`] ?? null;
}

export function getTerrainSprite(type: string): HTMLImageElement | null {
  return cache[`t:${type}`] ?? null;
}

export function isReady(img: HTMLImageElement | null): boolean {
  return !!img && img.complete && img.naturalWidth > 0;
}

export function onSpritesLoad(cb: () => void): void {
  reloadCallback = cb;
}
