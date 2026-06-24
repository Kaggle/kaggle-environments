// Sprite assets for the Reinforce Tactics visualizer.
//
// Two interchangeable art sets are bundled (Vite inlines them as data
// URLs via vite-plugin-singlefile):
//
//   - "game"   (./assets/sprites/game/): pixel art from the main
//     Reinforce Tactics repository. Units use the first idle frame of
//     their animation sheet. Units and capturable structures are
//     team-coloured at load time with the same palette swap the game
//     uses (base blue palette -> red for Player 1, base blue kept for
//     Player 2, gray for neutral structures).
//   - "kaggle" (./assets/sprites/): the original placeholder art that
//     shipped with this environment. Ownership is conveyed by the
//     renderer (discs / tints), not by the sprite itself.
//
// The active set is a runtime toggle (see set/getSpriteTheme); the
// renderer exposes it as a button in the status bar.

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

import gameWarriorUrl from './assets/sprites/game/warrior.png';
import gameMageUrl from './assets/sprites/game/mage.png';
import gameClericUrl from './assets/sprites/game/cleric.png';
import gameArcherUrl from './assets/sprites/game/archer.png';
import gameKnightUrl from './assets/sprites/game/knight.png';
import gameRogueUrl from './assets/sprites/game/rogue.png';
import gameSorcererUrl from './assets/sprites/game/sorcerer.png';
import gameBarbarianUrl from './assets/sprites/game/barbarian.png';
import gameHqUrl from './assets/sprites/game/headquarters.png';
import gameBuildingUrl from './assets/sprites/game/building.png';
import gameTowerUrl from './assets/sprites/game/tower.png';
import gameGrassUrl from './assets/sprites/game/grass.png';
import gameForestUrl from './assets/sprites/game/forest.png';
import gameMountainUrl from './assets/sprites/game/mountain.png';
import gameWaterUrl from './assets/sprites/game/water.png';
import gameOceanUrl from './assets/sprites/game/ocean.png';
import gameRoadUrl from './assets/sprites/game/road.png';

export type SpriteTheme = 'game' | 'kaggle';

// Engine codes -> sprite URL, per art set.
const UNIT_URLS: Record<SpriteTheme, Record<string, string>> = {
  kaggle: {
    W: warriorUrl,
    M: mageUrl,
    C: clericUrl,
    A: archerUrl,
    K: knightUrl,
    R: rogueUrl,
    S: sorcererUrl,
    B: barbarianUrl,
  },
  game: {
    W: gameWarriorUrl,
    M: gameMageUrl,
    C: gameClericUrl,
    A: gameArcherUrl,
    K: gameKnightUrl,
    R: gameRogueUrl,
    S: gameSorcererUrl,
    B: gameBarbarianUrl,
  },
};

const STRUCT_URLS: Record<SpriteTheme, Record<string, string>> = {
  kaggle: {
    h: hqUrl,
    b: buildingUrl,
    t: towerUrl,
  },
  game: {
    h: gameHqUrl,
    b: gameBuildingUrl,
    t: gameTowerUrl,
  },
};

// Terrain codes match the engine's TileType values.
const TERRAIN_URLS: Record<SpriteTheme, Record<string, string>> = {
  kaggle: {
    p: grassUrl,
    f: forestUrl,
    m: mountainUrl,
    w: waterUrl,
    o: oceanUrl,
    r: roadUrl,
  },
  game: {
    p: gameGrassUrl,
    f: gameForestUrl,
    m: gameMountainUrl,
    w: gameWaterUrl,
    o: gameOceanUrl,
    r: gameRoadUrl,
  },
};

// ---------------------------------------------------------------------------
// Team colour palettes (mirrors constants.py in the main repository)
// ---------------------------------------------------------------------------

type RGB = [number, number, number];

// Base sprite palette (blue tones) baked into the game art, from darkest
// to lightest. These exact pixels represent a unit's / structure's "team
// colour" regions and are replaced per owner below.
const BASE_SPRITE_COLORS: RGB[] = [
  [30, 87, 156], // #1e579c - darkest
  [60, 94, 139], // #3c5e8b
  [47, 114, 144], // #2f7290
  [40, 134, 176], // #2886b0
  [61, 165, 211], // #3da5d3 - lightest
  // Additional blue tones for units and bases
  [7, 109, 191], // #076dbf
  [0, 152, 219], // #0098db
  [79, 143, 186], // #4f8fba
  [115, 190, 211], // #73bed3
];

// Per-owner replacement palettes (same length / order as
// BASE_SPRITE_COLORS). `null` means "keep the base colours as-is" — the
// art is already blue, which is Player 2's colour in the main game.
const TEAM_PALETTES: Record<number, RGB[] | null> = {
  1: [
    // Red (Player 1)
    [156, 47, 38],
    [139, 72, 65],
    [168, 62, 54],
    [194, 58, 48],
    [220, 90, 75],
    // Replacements for additional blue tones
    [202, 31, 15],
    [232, 28, 9],
    [197, 97, 88],
    [224, 134, 126],
  ],
  2: null, // Blue (Player 2) – sprites are already blue, no swap needed
};

// Neutral (unowned) structure palette – white/gray tones used when a
// capturable structure has no owning player (owner === 0).
const NEUTRAL_STRUCTURE_PALETTE: RGB[] = [
  [100, 100, 112], // dark gray
  [120, 120, 130], // medium-dark gray
  [142, 142, 152], // medium gray
  [172, 172, 182], // medium-light gray
  [204, 204, 214], // light gray
  // Neutrals for additional blue tones
  [174, 174, 185],
  [200, 200, 212],
  [170, 170, 180],
  [192, 192, 205],
];

// ---------------------------------------------------------------------------
// Theme state
// ---------------------------------------------------------------------------

const THEME_STORAGE_KEY = 'reinforce-tactics-sprite-theme';

function loadStoredTheme(): SpriteTheme {
  try {
    const stored = window.localStorage?.getItem(THEME_STORAGE_KEY);
    if (stored === 'game' || stored === 'kaggle') return stored;
  } catch {
    // Sandboxed iframes may block storage access; fall through.
  }
  return 'game';
}

let activeTheme: SpriteTheme = loadStoredTheme();

export function getSpriteTheme(): SpriteTheme {
  return activeTheme;
}

export function setSpriteTheme(theme: SpriteTheme): void {
  activeTheme = theme;
  try {
    window.localStorage?.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Persistence is best-effort only.
  }
}

// ---------------------------------------------------------------------------
// Loading & recolouring
// ---------------------------------------------------------------------------

const imageCache: Record<string, HTMLImageElement> = {};

// Recoloured variants, built lazily once the base image has loaded.
// Key: `${imageKey}|${owner}`.
const recolorCache: Record<string, HTMLCanvasElement> = {};

// Callback fired the first time any pending sprite finishes loading,
// so the renderer can request a redraw without polling.
let reloadCallback: (() => void) | null = null;

function preload(key: string, url: string): HTMLImageElement {
  if (imageCache[key]) return imageCache[key];
  const img = new Image();
  img.onload = () => {
    if (reloadCallback) reloadCallback();
  };
  img.src = url;
  imageCache[key] = img;
  return img;
}

// Kick off all loads immediately at module import.
for (const theme of ['kaggle', 'game'] as SpriteTheme[]) {
  for (const [k, u] of Object.entries(UNIT_URLS[theme])) preload(`${theme}:u:${k}`, u);
  for (const [k, u] of Object.entries(STRUCT_URLS[theme])) preload(`${theme}:s:${k}`, u);
  for (const [k, u] of Object.entries(TERRAIN_URLS[theme])) preload(`${theme}:t:${k}`, u);
}

function imageLoaded(img: HTMLImageElement | undefined): img is HTMLImageElement {
  return !!img && img.complete && img.naturalWidth > 0;
}

/**
 * Replace base palette colours with an owner's palette in a sprite.
 *
 * Exact-match pixel replacement, the canvas equivalent of the
 * pygame.PixelArray.replace swap the main repository performs at sprite
 * load time. Returns a canvas usable as a drawImage source.
 */
function recolorSprite(img: HTMLImageElement, palette: RGB[]): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // Pack the base palette into a lookup of rgb -> replacement.
  const lookup = new Map<number, RGB>();
  for (let i = 0; i < BASE_SPRITE_COLORS.length && i < palette.length; i++) {
    const [r, g, b] = BASE_SPRITE_COLORS[i];
    lookup.set((r << 16) | (g << 8) | b, palette[i]);
  }

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    const key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    const repl = lookup.get(key);
    if (repl) {
      data[i] = repl[0];
      data[i + 1] = repl[1];
      data[i + 2] = repl[2];
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/**
 * Resolve the owner-coloured variant of a game-art sprite.
 *
 * Owner 1 gets the red palette, owner 2 keeps the base blue art, and
 * any other owner (0 = neutral) gets the gray structure palette when
 * `neutralFallback` is set (structures) or the base art otherwise
 * (units never render unowned).
 */
function getOwnerVariant(imageKey: string, owner: number, neutralFallback: boolean): CanvasImageSource | null {
  const img = imageCache[imageKey];
  if (!imageLoaded(img)) return null;

  let palette: RGB[] | null;
  if (owner === 1 || owner === 2) {
    palette = TEAM_PALETTES[owner];
  } else {
    palette = neutralFallback ? NEUTRAL_STRUCTURE_PALETTE : null;
  }
  if (palette === null) return img;

  const cacheKey = `${imageKey}|${owner}`;
  let canvas = recolorCache[cacheKey];
  if (!canvas) {
    canvas = recolorSprite(img, palette);
    recolorCache[cacheKey] = canvas;
  }
  return canvas;
}

// ---------------------------------------------------------------------------
// Public lookups
// ---------------------------------------------------------------------------

export function getUnitSprite(type: string, owner = 0): CanvasImageSource | null {
  if (activeTheme === 'game') {
    if (!(type in UNIT_URLS.game)) return null;
    return getOwnerVariant(`game:u:${type}`, owner, false);
  }
  const img = imageCache[`kaggle:u:${type}`];
  return imageLoaded(img) ? img : null;
}

export function getStructureSprite(type: string, owner = 0): CanvasImageSource | null {
  if (activeTheme === 'game') {
    if (!(type in STRUCT_URLS.game)) return null;
    return getOwnerVariant(`game:s:${type}`, owner, true);
  }
  const img = imageCache[`kaggle:s:${type}`];
  return imageLoaded(img) ? img : null;
}

export function getTerrainSprite(type: string): CanvasImageSource | null {
  const img = imageCache[`${activeTheme}:t:${type}`];
  return imageLoaded(img) ? img : null;
}

export function isReady(source: CanvasImageSource | null): boolean {
  if (!source) return false;
  if (source instanceof HTMLImageElement) {
    return source.complete && source.naturalWidth > 0;
  }
  return true;
}

export function onSpritesLoad(cb: () => void): void {
  reloadCallback = cb;
}
