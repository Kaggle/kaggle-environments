import { Container, Graphics, Sprite, Text, TextStyle, type Spritesheet } from 'pixi.js';
import { BOARD_PADDING, BOARD_PX, LINE_COLOR, getStarPoints, getCellSize, gridToPixel } from './constants.ts';

const TILE_SIZE = 128;
const TILE_MARGIN = 2;
const TILE_INNER = TILE_SIZE - 2 * TILE_MARGIN;

const STAR_RADIUS_RATIO = 0.08;

const COL_LETTERS = 'ABCDEFGHJKLMNOPQRST';
const LABEL_COLOR = 0x000000;
const LABEL_MAX_FONT_SIZE = 11;
const LABEL_FONT_SIZE_RATIO = 0.38;
const LABEL_OFFSET_RATIO = 0.45;

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
  const starRadius = cell * STAR_RADIUS_RATIO;
  const stars = getStarPoints(boardSize);
  const sg = new Graphics();
  for (const [row, col] of stars) {
    const { x, y } = gridToPixel(row, col, boardSize);
    sg.circle(x, y, starRadius);
  }
  sg.fill(LINE_COLOR);
  container.addChild(sg);

  // Row & column labels
  const labelOffset = BOARD_PADDING * LABEL_OFFSET_RATIO;
  const labelStyle = new TextStyle({
    fontFamily: 'sans-serif',
    fontSize: Math.min(LABEL_MAX_FONT_SIZE, BOARD_PADDING * LABEL_FONT_SIZE_RATIO),
    fill: LABEL_COLOR,
  });

  // Column letters (A–T, skipping I) — top and bottom
  for (let col = 0; col < boardSize; col++) {
    const { x } = gridToPixel(0, col, boardSize);
    for (const y of [labelOffset, BOARD_PX - labelOffset]) {
      const label = new Text({ text: COL_LETTERS[col], style: labelStyle });
      label.anchor.set(0.5);
      label.position.set(x, y);
      container.addChild(label);
    }
  }

  // Row numbers (1 at bottom, N at top) — left and right
  for (let row = 0; row < boardSize; row++) {
    const { y } = gridToPixel(row, 0, boardSize);
    for (const x of [labelOffset, BOARD_PX - labelOffset]) {
      const label = new Text({ text: String(boardSize - row), style: labelStyle });
      label.anchor.set(0.5);
      label.position.set(x, y);
      container.addChild(label);
    }
  }

  container.cacheAsTexture(true);

  return container;
}
