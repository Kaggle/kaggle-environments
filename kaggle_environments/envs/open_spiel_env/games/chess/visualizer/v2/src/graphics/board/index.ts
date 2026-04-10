import { Graphics } from 'pixi.js';
import { BOARD_SIZE } from '../constants';

const LIGHT_SQUARE_COLOR = 0xffffff;
const DARK_SQUARE_COLOR = 0x000000;

// TODO(pim-at-stink): Use textures for the board.
export function drawBoard(squareSize: number): Graphics {
  const g = new Graphics();

  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const isLight = (row + col) % 2 === 0;
      g.rect(col * squareSize, row * squareSize, squareSize, squareSize).fill(
        isLight ? LIGHT_SQUARE_COLOR : DARK_SQUARE_COLOR
      );
    }
  }

  g.alpha = 0.6;

  return g;
}
