export type CellValue = '.' | 'B' | 'W';

export interface GridPos {
  row: number;
  col: number;
}

export interface Captures {
  black: number;
  white: number;
}

export interface Territory {
  black: GridPos[];
  white: GridPos[];
}
