import { Chess } from 'chess.js';

export enum HeroTypes {
  CHECKMATE,
  QUEEN_LOSS,
  PROMOTION,
  CASTLING,
  FIRST_CAPTURE,
}

export function detectHeroType(game: Chess): HeroTypes | null {
  if (game === null) return null;
  return null;
}
