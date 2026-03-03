import replay from './replay.json';

export type CellValue = '.' | 'B' | 'W'

interface BoardState {
  board_size: number;
  board_grid: Record<string, CellValue>[][];
}

const states: BoardState[] = (replay.info.stateHistory as string[]).map(
    s => JSON.parse(s) as BoardState
);

export const boardSize = states[0].board_size;
export const totalSteps = states.length;

// Precompute all grids once
const grids: CellValue[][][] = states.map(state =>
    state.board_grid.map(row =>
        row.map(cell => Object.values(cell)[0] as CellValue)
    )
);

// Precompute last played position for each step
const lastPlayedMoves: ({ row: number; col: number } | null)[] = grids.map((grid, i) => {
  if (i === 0) return null;
  const prev = grids[i - 1];
  for (let row = 0; row < grid.length; row++) {
    for (let col = 0; col < grid[row].length; col++) {
      if (prev[row][col] === '.' && grid[row][col] !== '.') {
        return { row, col };
      }
    }
  }
  return null;
});

export function getGrid(step: number): CellValue[][] {
  return grids[step];
}

export function getLastPlayed(step: number): { row: number; col: number } | null {
  console.log(lastPlayedMoves[step]);
  return lastPlayedMoves[step];
}

// Precompute cumulative capture counts per step
export interface Captures {
  black: number;
  white: number;
}

const capturedCounts: Captures[] = [{ black: 0, white: 0 }];
for (let i = 1; i < grids.length; i++) {
  let { black, white } = capturedCounts[i - 1];
  const prev = grids[i - 1];
  const curr = grids[i];
  for (let r = 0; r < curr.length; r++) {
    for (let c = 0; c < curr[r].length; c++) {
      if (prev[r][c] === 'B' && curr[r][c] === '.') black++;
      else if (prev[r][c] === 'W' && curr[r][c] === '.') white++;
    }
  }
  capturedCounts.push({ black, white });
}

export function getCaptures(step: number): Captures {
  return capturedCounts[step];
}
