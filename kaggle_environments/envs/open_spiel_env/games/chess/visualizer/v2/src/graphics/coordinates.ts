import { BOARD_SIZE, CHAR_CODE_A } from './constants';

export type Orientation = 'white' | 'black';

/**
 * Convert a square (e.g. "e4") to a pixel coordinate.
 *
 * Row 1 is White's back row (screen-bottom), row 8 is Black's (screen-top).
 *
 * Because screen Y grows downward but chess rows grow upward, we flip the row
 * index when rendering from White's perspective. Black's perspective flips both axes.
 */
export function squareToPixel(
  square: string,
  squareSize: number,
  orientation: Orientation,
  offset: number = 0
): { x: number; y: number } {
  const boardCol = square.charCodeAt(0) - CHAR_CODE_A;
  const boardRow = Number(square[1]) - 1;

  const screenCol = orientation === 'white' ? boardCol : BOARD_SIZE - 1 - boardCol;
  const screenRow = orientation === 'white' ? BOARD_SIZE - 1 - boardRow : boardRow;

  return {
    x: offset + screenCol * squareSize + squareSize / 2,
    y: offset + screenRow * squareSize + squareSize / 2,
  };
}
