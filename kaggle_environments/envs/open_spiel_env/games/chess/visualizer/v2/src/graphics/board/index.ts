import { Container, Graphics, Sprite, Text, TextStyle, type Texture } from 'pixi.js';
import { BOARD_SIZE } from '../constants';

const COLUMN_LETTERS = 'ABCDEFGH';
const LABEL_COLOR = 0x000000;
const LABEL_MAX_FONT_SIZE = 20;
const LABEL_FONT_SIZE_RATIO = 0.5;

export function drawBoard(squareSize: number, boardOffset: number, tileTexture: Texture): Container {
  const container = new Container();

  // Place black-tile sprites on every dark square. Light squares are
  // transparent.
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const isDark = (row + col) % 2 !== 0;
      if (!isDark) continue;

      const sprite = new Sprite({
        texture: tileTexture,
        anchor: 0.5,
        width: squareSize,
        height: squareSize,
        x: boardOffset + col * squareSize + squareSize / 2,
        y: boardOffset + row * squareSize + squareSize / 2,
      });
      container.addChild(sprite);
    }
  }

  // 1px grid lines
  const gridLines = new Graphics();
  const gridStart = boardOffset;
  const gridEnd = boardOffset + BOARD_SIZE * squareSize;

  for (let i = 0; i <= BOARD_SIZE; i++) {
    // +0.5 to avoid subpixeling.
    const pos = Math.round(boardOffset + i * squareSize) + 0.5;
    // Columns
    gridLines.moveTo(gridStart, pos).lineTo(gridEnd, pos);
    // Rows
    gridLines.moveTo(pos, gridStart).lineTo(pos, gridEnd);
  }
  gridLines.stroke({ width: 1, color: 0x000000 });
  container.addChild(gridLines);

  // Labels
  const fontSize = Math.min(LABEL_MAX_FONT_SIZE, squareSize * LABEL_FONT_SIZE_RATIO);
  const labelStyle = new TextStyle({
    fontFamily: '"Inter", sans-serif',
    fontSize,
    fill: LABEL_COLOR,
  });

  const boardEnd = boardOffset + BOARD_SIZE * squareSize;
  const labelMarginCenter = boardOffset / 2;

  // Column letters.
  for (let col = 0; col < BOARD_SIZE; col++) {
    const x = boardOffset + col * squareSize + squareSize / 2;
    for (const y of [labelMarginCenter, boardEnd + labelMarginCenter]) {
      const label = new Text({ text: COLUMN_LETTERS[col], style: labelStyle });
      label.anchor.set(0.5);
      label.position.set(Math.round(x), Math.round(y));
      container.addChild(label);
    }
  }

  // Row numbers.
  for (let row = 0; row < BOARD_SIZE; row++) {
    const y = boardOffset + row * squareSize + squareSize / 2;
    for (const x of [labelMarginCenter, boardEnd + labelMarginCenter]) {
      const label = new Text({ text: String(BOARD_SIZE - row), style: labelStyle });
      label.anchor.set(0.5);
      label.position.set(Math.round(x), Math.round(y));
      container.addChild(label);
    }
  }

  return container;
}
