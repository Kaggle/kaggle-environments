import type { CellValue } from '../../replay/middleman.ts';

export interface GridDiff {
  added: { row: number; col: number; value: 'B' | 'W' }[];
  removed: { row: number; col: number }[];
}

export function diffGrids(
  prev: CellValue[][] | null,
  next: CellValue[][],
): GridDiff {
  const added: GridDiff['added'] = [];
  const removed: GridDiff['removed'] = [];

  for (let row = 0; row < next.length; row++) {
    for (let col = 0; col < next[row].length; col++) {
      const prevVal = prev ? prev[row][col] : '.';
      const nextVal = next[row][col];

      if (prevVal === '.' && nextVal !== '.') {
        added.push({ row, col, value: nextVal });
      } else if (prevVal !== '.' && nextVal === '.') {
        removed.push({ row, col });
      } else if (prevVal !== '.' && nextVal !== '.' && prevVal !== nextVal) {
        removed.push({ row, col });
        added.push({ row, col, value: nextVal });
      }
    }
  }

  return { added, removed };
}
