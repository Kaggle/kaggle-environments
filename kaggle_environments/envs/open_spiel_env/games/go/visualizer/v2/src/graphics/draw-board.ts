import { Container, Sprite, Text, TextStyle, TilingSprite, type Spritesheet } from 'pixi.js';
import { BOARD_PADDING, BOARD_PX, getStarPoints, getCellSize, gridToPixel } from './constants.ts';

const HOSHI_SIZE_RATIO = 0.25;

const COL_LETTERS = 'ABCDEFGHJKLMNOPQRST';
const LABEL_COLOR = 0x000000;
const LABEL_MAX_FONT_SIZE = 11;
const LABEL_FONT_SIZE_RATIO = 0.38;
const LABEL_OFFSET_RATIO = 0.45;

export function drawBoard(boardSize: number, sheet: Spritesheet): Container {
  const container = new Container();
  const cell = getCellSize(boardSize);

  const gridSpan = (boardSize - 1) * cell;
  const lineTexture = sheet.textures['squiggle-dash.png'];
  const lineHeight = lineTexture.height / 2;

  for (let row = 0; row < boardSize; row++) {
    const { x, y } = gridToPixel(row, 0, boardSize);
    const line = new TilingSprite({
      texture: lineTexture,
      width: gridSpan,
      height: lineHeight,
      applyAnchorToTexture: true,
    });
    line.clampMargin = 0;
    line.tileScale.set(0.5);
    line.anchor.set(0, 0.5);
    line.position.set(x, y);
    container.addChild(line);
  }

  for (let col = 0; col < boardSize; col++) {
    const { x, y } = gridToPixel(0, col, boardSize);
    const line = new TilingSprite({
      texture: lineTexture,
      width: gridSpan,
      height: lineHeight,
      applyAnchorToTexture: true,
    });
    line.clampMargin = 0;
    line.tileScale.set(0.5);
    line.anchor.set(0, 0.5);
    line.position.set(x, y);
    line.angle = 90;
    container.addChild(line);
  }

  // Star points (hoshi)
  const hoshiSize = cell * HOSHI_SIZE_RATIO;
  for (const [row, col] of getStarPoints(boardSize)) {
    const { x, y } = gridToPixel(row, col, boardSize);
    const hoshi = new Sprite(sheet.textures['hoshi.png']);
    hoshi.anchor.set(0.5);
    hoshi.position.set(x, y);
    hoshi.width = hoshiSize;
    hoshi.height = hoshiSize;
    container.addChild(hoshi);
  }

  // Row & column labels
  const labelOffset = BOARD_PADDING * LABEL_OFFSET_RATIO;
  const labelStyle = new TextStyle({
    fontFamily: '"Google Sans", sans-serif',
    fontSize: Math.min(LABEL_MAX_FONT_SIZE, BOARD_PADDING * LABEL_FONT_SIZE_RATIO),
    fill: LABEL_COLOR,
  });

  // Column letters (A–T, skipping I) — top and bottom
  for (let col = 0; col < boardSize; col++) {
    const { x } = gridToPixel(0, col, boardSize);
    for (const y of [labelOffset, BOARD_PX - labelOffset]) {
      const label = new Text({ text: COL_LETTERS[col], style: labelStyle });
      label.anchor.set(0.5);
      label.position.set(Math.round(x), Math.round(y));
      container.addChild(label);
    }
  }

  // Row numbers (1 at bottom, N at top) — left and right
  for (let row = 0; row < boardSize; row++) {
    const { y } = gridToPixel(row, 0, boardSize);
    for (const x of [labelOffset, BOARD_PX - labelOffset]) {
      const label = new Text({ text: String(boardSize - row), style: labelStyle });
      label.anchor.set(0.5);
      label.position.set(Math.round(x), Math.round(y));
      container.addChild(label);
    }
  }

  container.cacheAsTexture({ resolution: window.devicePixelRatio });

  return container;
}
