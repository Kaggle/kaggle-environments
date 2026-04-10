export const BOARD_SIZE = 8;
export const CHAR_CODE_A = 97; // 'a'.charCodeAt(0)

export const LAYERS = ['background', 'pieces'] as const;
export type Layer = (typeof LAYERS)[number];

export type PieceType = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
export type PieceColor = 'w' | 'b';

export const PIECE_NAMES: Record<PieceType, string> = {
  p: 'pawn',
  n: 'knight',
  b: 'bishop',
  r: 'rook',
  q: 'queen',
  k: 'king',
};
