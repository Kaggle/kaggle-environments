export interface MancalaObservation {
  board: number[];
  pits: { '0': number[]; '1': number[] };
  stores: { '0': number; '1': number };
  scores: [number, number];
  current_player: number | string;
  move_number: number;
  last_action: number | null;
  is_terminal: boolean;
  winner: number | string | null;
}

// Index conventions for the visual layout. OpenSpiel mancala stores 14 cells:
//   board[0]      = player 1's store (rendered LEFT)
//   board[1..6]   = player 0's pits  (rendered BOTTOM, left-to-right)
//   board[7]      = player 0's store (rendered RIGHT)
//   board[8..13]  = player 1's pits  (rendered TOP, RIGHT-to-LEFT)
export const STORE_LEFT = 0; // player 1
export const STORE_RIGHT = 7; // player 0
export const BOTTOM_ROW = [1, 2, 3, 4, 5, 6]; // player 0, left-to-right
export const TOP_ROW = [13, 12, 11, 10, 9, 8]; // player 1, left-to-right (sowing CCW)

// Path tint: pits closest to the source are green; the hue slides toward blue as
// the seed continues along the sowing path, landing fully blue at the last dest.
const PATH_START = [80, 180, 110]; // green
const PATH_END = [0, 138, 187]; // blue
export function pathTint(step: number, total: number, alpha = 0.32): string {
  const t = total <= 1 ? 1 : (step - 1) / (total - 1);
  const c = PATH_START.map((s, i) => Math.round(s + (PATH_END[i] - s) * t));
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
}
