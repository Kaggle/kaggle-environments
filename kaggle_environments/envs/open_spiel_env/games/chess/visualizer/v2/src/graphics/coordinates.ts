import { BOARD_SIZE, CHAR_CODE_A } from './constants';

export type Orientation = 'white' | 'black';

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
