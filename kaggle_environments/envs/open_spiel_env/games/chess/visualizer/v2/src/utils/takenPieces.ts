import { Chess, Color, PieceSymbol } from 'chess.js';

type TakenPieces = { color: Color; type: PieceSymbol }[];

export function takenPieces(game: Chess): TakenPieces {
  const takenPieces: TakenPieces = [];

  for (const move of game.history({ verbose: true })) {
    if (!move.captured) continue;
    const opponent = move.color === 'b' ? 'w' : 'b';
    takenPieces.push({ color: opponent, type: move.captured });
  }

  return takenPieces;
}
