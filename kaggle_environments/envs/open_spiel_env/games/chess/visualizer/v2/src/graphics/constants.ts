export const BOARD_SIZE = 8;
export const CHAR_CODE_A = 97; // 'a'.charCodeAt(0)

/**
 * Padding around the 8x8 grid as a percentage of one square. Used to position the board labels.
 * @see drawBoard
 */
export const BOARD_PADDING_RATIO = 0.4;

export const GRID_LINE_WIDTH = 1;

export const LAYERS = ['background', 'highlights', 'pieces'] as const;
export type Layer = (typeof LAYERS)[number];

export type PieceType = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
export type PieceColor = 'w' | 'b';
