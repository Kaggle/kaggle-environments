import { Chess } from 'chess.js';

export enum HeroTypes {
  CHECKMATE,
  QUEEN_LOSS,
  PROMOTION,
  CASTLING,
  FIRST_CAPTURE,
}

export function detectHeroType(game: Chess): HeroTypes | null {
  const move = game.history({ verbose: true }).at(-1);
  const pieceCount = game
    .board()
    .flat()
    .filter((p) => p !== null).length;

  const isCheckmate = game.isCheckmate();
  const isQueenLoss = move?.captured === 'q';
  const isPromotion = move?.isPromotion();
  const isCastling = move?.isKingsideCastle() || move?.isQueensideCastle();
  const isFirstCapture = move?.isCapture() && pieceCount === 31;

  if (isCheckmate) return HeroTypes.CHECKMATE;
  if (isQueenLoss) return HeroTypes.QUEEN_LOSS;
  if (isPromotion) return HeroTypes.PROMOTION;
  if (isCastling) return HeroTypes.CASTLING;
  if (isFirstCapture) return HeroTypes.FIRST_CAPTURE;

  return null;
}
