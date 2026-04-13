import { Container, Graphics, Text, TextStyle } from 'pixi.js';
import { BOARD_SIZE } from '../constants';

const LIGHT_SQUARE_COLOR = 0xffffff;
const DARK_SQUARE_COLOR = 0x000000;

const FILE_LETTERS = 'ABCDEFGH';
const LABEL_COLOR = 0x000000;
const LABEL_MAX_FONT_SIZE = 20;
const LABEL_FONT_SIZE_RATIO = 0.5;

// TODO(pim-at-stink): Use textures for the board.
export function drawBoard(squareSize: number, boardOffset: number): Container {
  const container = new Container();

  const g = new Graphics();
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const isLight = (row + col) % 2 === 0;
      const x = boardOffset + col * squareSize;
      const y = boardOffset + row * squareSize;
      g.rect(x, y, squareSize, squareSize).fill(isLight ? LIGHT_SQUARE_COLOR : DARK_SQUARE_COLOR);
    }
  }
  g.alpha = 0.6;
  container.addChild(g);

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
      const label = new Text({ text: FILE_LETTERS[col], style: labelStyle });
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
