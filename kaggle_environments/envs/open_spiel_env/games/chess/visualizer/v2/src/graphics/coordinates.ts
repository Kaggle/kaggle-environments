import { BOARD_SIZE, CHAR_CODE_A } from './constants';

export type Orientation = 'white' | 'black';

export function squareToPixel(square: string, squareSize: number, orientation: Orientation): { x: number; y: number } {
  const file = square.charCodeAt(0) - CHAR_CODE_A;
  const rank = Number(square[1]) - 1;

  const col = orientation === 'white' ? file : BOARD_SIZE - 1 - file;
  const row = orientation === 'white' ? BOARD_SIZE - 1 - rank : rank;

  return {
    x: col * squareSize + squareSize / 2,
    y: row * squareSize + squareSize / 2,
  };
}
