import type { GridPos } from '../types/game.ts';

export const BOARD_PX = 512;
export const BOARD_PADDING = 28;

export const LINE_COLOR = 0x1a1a1a;
export const LINE_WIDTH = 1;

/** Star point (hoshi) positions for a 9×9 board */
export const STAR_POINTS_9: [number, number][] = [
  [2, 2],
  [2, 6],
  [4, 4],
  [6, 2],
  [6, 6],
];

export function getCellSize(boardSize: number): number {
  return (BOARD_PX - BOARD_PADDING * 2) / (boardSize - 1);
}

/** Map a board intersection (row, col) to pixel coordinates */
export function gridToPixel(row: number, col: number, boardSize: number): { x: number; y: number } {
  const cell = getCellSize(boardSize);
  return {
    x: BOARD_PADDING + col * cell,
    y: BOARD_PADDING + row * cell,
  };
}

// Capture pot area below the board
export const POT_AREA_HEIGHT = 150;
export const POT_SIZE = 120;
export const POT_PRISONER_SIZE = 32;
export const POT_SCATTER_RADIUS = 30;
export const POT_MAX_PRISONERS = 30;

const NEIGHBOR_DELTAS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
] as const;

/** Return orthogonal neighbors within board bounds */
export function getNeighbors(row: number, col: number, boardSize: number): GridPos[] {
  const neighbors: GridPos[] = [];
  for (const [dr, dc] of NEIGHBOR_DELTAS) {
    const r = row + dr;
    const c = col + dc;
    if (r >= 0 && r < boardSize && c >= 0 && c < boardSize) {
      neighbors.push({ row: r, col: c });
    }
  }
  return neighbors;
}
