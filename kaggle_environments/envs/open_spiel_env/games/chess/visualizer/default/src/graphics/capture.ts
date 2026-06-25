import { Sprite } from 'pixi.js';
import type { Chess } from 'chess.js';
import { animate } from 'motion';
import type { Engine } from './engine';
import { squareToPixel } from './coordinates';

const PIECE_FADE_DURATION = 0.3; // seconds — piece fades and shrinks
const PIECE_SCALE_TARGET = 0.6; // piece shrinks to 60% of original

export function syncCapture(engine: Engine, chess: Chess, snap: boolean) {
  if (snap) return;

  const lastMove = chess.history({ verbose: true }).at(-1);
  if (!lastMove?.captured) return;

  const { squareSize, textures, resources, boardOffset } = engine;

  const capturedColor = lastMove.color === 'w' ? 'b' : 'w';
  const texture = textures[`${capturedColor}${lastMove.captured}.png`];
  if (!texture) return;

  // En passant: captured pawn sits on the target file but the source rank.
  const captureSquare = lastMove.isEnPassant() ? `${lastMove.to[0]}${lastMove.from[1]}` : lastMove.to;

  const pos = squareToPixel(captureSquare, squareSize, 'white', boardOffset);

  const piece = new Sprite({ texture, anchor: 0.5 });
  piece.scale.set(squareSize / texture.width);
  piece.position.set(pos.x, pos.y);
  resources.vfx.addChild(piece);

  const pieceScale = squareSize / texture.width;
  engine.animations.add(
    animate(
      piece.scale,
      { x: pieceScale * PIECE_SCALE_TARGET, y: pieceScale * PIECE_SCALE_TARGET },
      { duration: PIECE_FADE_DURATION }
    )
  );
  engine.animations.add(animate(piece, { alpha: 0 }, { duration: PIECE_FADE_DURATION }));
}
