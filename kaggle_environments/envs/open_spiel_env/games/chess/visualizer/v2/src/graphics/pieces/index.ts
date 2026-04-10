import { Assets, Sprite, type Texture } from 'pixi.js';
import { Chess } from 'chess.js';
import type { Engine } from '../engine';
import { squareToPixel } from '../coordinates';
import type { PieceColor, PieceType } from '../constants';

import bishopBlackPath from '../../assets/images/bishop-b-small.webp';
import bishopWhitePath from '../../assets/images/bishop-w-small.webp';
import kingBlackPath from '../../assets/images/king-b-small.webp';
import kingWhitePath from '../../assets/images/king-w-small.webp';
import knightBlackPath from '../../assets/images/knight-b-small.webp';
import knightWhitePath from '../../assets/images/knight-w-small.webp';
import pawnBlackPath from '../../assets/images/pawn-b-small.webp';
import pawnWhitePath from '../../assets/images/pawn-w-small.webp';
import queenBlackPath from '../../assets/images/queen-b-small.webp';
import queenWhitePath from '../../assets/images/queen-w-small.webp';
import rookBlackPath from '../../assets/images/rook-b-small.webp';
import rookWhitePath from '../../assets/images/rook-w-small.webp';

const PIECE_PATHS: Record<`${PieceColor}${PieceType}`, string> = {
  wp: pawnWhitePath,
  wn: knightWhitePath,
  wb: bishopWhitePath,
  wr: rookWhitePath,
  wq: queenWhitePath,
  wk: kingWhitePath,
  bp: pawnBlackPath,
  bn: knightBlackPath,
  bb: bishopBlackPath,
  br: rookBlackPath,
  bq: queenBlackPath,
  bk: kingBlackPath,
};

export async function loadPieceTextures(): Promise<Record<string, Texture>> {
  const entries = Object.entries(PIECE_PATHS);
  const loaded = await Promise.all(entries.map(async ([key, path]) => [key, await Assets.load(path)] as const));
  return Object.fromEntries(loaded);
}

// TODO(pim-at-stink): Note - We may need to make these sprites persist
//  throughout the game.
export function syncPieces(engine: Engine, chess: Chess) {
  const { squareSize, textures, resources } = engine;

  resources.pieces.removeChildren();

  const board = chess.board();

  for (const row of board) {
    for (const cell of row) {
      if (!cell) continue;

      const texture = textures[`${cell.color}${cell.type}`];
      if (!texture) continue;

      const sprite = new Sprite({ texture, anchor: 0.5 });
      sprite.scale.set(squareSize / texture.width);

      const { x, y } = squareToPixel(cell.square, squareSize, 'white');
      sprite.position.set(x, y);

      resources.pieces.addChild(sprite);
    }
  }
}
