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
