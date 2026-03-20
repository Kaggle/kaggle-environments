import { Sprite, type Spritesheet } from 'pixi.js';
import type { CellValue } from '../types/game.ts';
import { getCellSize, getStoneScale, gridToPixel } from './constants.ts';

export interface RestState {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
}

export interface StonePair {
  stone: Sprite;
  shadow: Sprite;
  value: 'B' | 'W';
  stoneRest: RestState;
  shadowRest: RestState;
}

export type StoneMap = Map<string, StonePair>;

export function resetPair(pair: StonePair): void {
  const { stone, shadow, stoneRest, shadowRest } = pair;
  stone.position.set(stoneRest.x, stoneRest.y);
  stone.scale.set(stoneRest.scaleX, stoneRest.scaleY);
  stone.alpha = 1;
  stone.rotation = 0;
  shadow.position.set(shadowRest.x, shadowRest.y);
  shadow.scale.set(shadowRest.scaleX, shadowRest.scaleY);
  shadow.alpha = 1;
}

export function posKey(row: number, col: number): string {
  return `${row},${col}`;
}

export function createStonePair(
  row: number,
  col: number,
  value: Exclude<CellValue, '.'>,
  boardSize: number,
  sheet: Spritesheet
): StonePair {
  const cell = getCellSize(boardSize);
  const stoneSize = cell * getStoneScale(boardSize);
  const shadowOffset = cell * 0.06;
  const { x, y } = gridToPixel(row, col, boardSize);

  const shadow = new Sprite(sheet.textures['shadow.png']);
  shadow.anchor.set(0.5);
  shadow.position.set(x - shadowOffset, y + shadowOffset);
  shadow.width = stoneSize;
  shadow.height = stoneSize;

  const texName = value === 'B' ? 'black.png' : 'white.png';
  const stone = new Sprite(sheet.textures[texName]);
  stone.anchor.set(0.5);
  stone.position.set(x, y);
  stone.width = stoneSize;
  stone.height = stoneSize;

  return {
    stone,
    shadow,
    value,
    stoneRest: { x, y, scaleX: stone.scale.x, scaleY: stone.scale.y },
    shadowRest: {
      x: x - shadowOffset,
      y: y + shadowOffset,
      scaleX: shadow.scale.x,
      scaleY: shadow.scale.y,
    },
  };
}
