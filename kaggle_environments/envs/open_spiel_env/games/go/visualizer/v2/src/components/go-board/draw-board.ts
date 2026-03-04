import { Container, Graphics, Sprite, type Spritesheet } from 'pixi.js';
import { LINE_COLOR, STAR_POINTS_9, getCellSize, gridToPixel } from './constants.ts';

const TILE_SIZE = 128;
const TILE_MARGIN = 2;
const TILE_INNER = TILE_SIZE - 2 * TILE_MARGIN; // 124

export function drawBoard(boardSize: number, sheet: Spritesheet): Container {
  const container = new Container();
  const cell = getCellSize(boardSize);

  // Tile each cell — anchor at margin boundary so strokes align with intersections
  const tileDisplaySize = (cell * TILE_SIZE) / TILE_INNER;
  for (let r = 0; r < boardSize - 1; r++) {
    for (let c = 0; c < boardSize - 1; c++) {
      const { x, y } = gridToPixel(r, c, boardSize);
      const tile = new Sprite(sheet.textures['board-tile.png']);
      tile.anchor.set(TILE_MARGIN / TILE_SIZE);
      tile.position.set(x, y);
      tile.width = tileDisplaySize;
      tile.height = tileDisplaySize;
      container.addChild(tile);
    }
  }

  // Star points (hoshi)
  const starRadius = cell * 0.08;
  const stars = boardSize === 9 ? STAR_POINTS_9 : [];
  const sg = new Graphics();
  for (const [row, col] of stars) {
    const { x, y } = gridToPixel(row, col, boardSize);
    sg.circle(x, y, starRadius);
  }
  sg.fill(LINE_COLOR);
  container.addChild(sg);

  return container;
}
