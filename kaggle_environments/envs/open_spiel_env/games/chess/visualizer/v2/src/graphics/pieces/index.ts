import { Assets, Sprite, type Texture } from 'pixi.js';
import { Chess } from 'chess.js';
import { animate, AnimationOptions } from 'motion';
import type { Engine } from '../engine';
import { squareToPixel } from '../coordinates';
import type { PieceColor, PieceType } from '../constants';
import { SCRUB_THRESHOLD_MS } from '../../constants';

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

// Motion config.
const SPRING_CONFIG: AnimationOptions = { type: 'spring' as const, stiffness: 180, damping: 22, mass: 1 };

// Get where every piece starts, and were every piece ends.
function getAnimationSources(chess: Chess): Map<string, string> | null {
  const lastMove = chess.history({ verbose: true }).at(-1);
  if (!lastMove) return null;

  const sources = new Map<string, string>([[lastMove.to, lastMove.from]]);

  // https://en.wikipedia.org/wiki/Castling#Description
  const row = lastMove.to[1];
  if (lastMove.isKingsideCastle()) sources.set(`f${row}`, `h${row}`);
  else if (lastMove.isQueensideCastle()) sources.set(`d${row}`, `a${row}`);

  return sources;
}

export function syncPieces(engine: Engine, chess: Chess, step: number) {
  const { squareSize, textures, resources } = engine;

  for (const anim of engine.animations) anim.stop();
  engine.animations.clear();

  // Snap when:
  //   1. The user went backwards.
  //   2. The user is scrubbing.
  const now = performance.now();
  const isFirstUpdate = engine.lastUpdateTime === 0;
  const timeSinceLastUpdate = now - engine.lastUpdateTime;
  const goingBackwards = step < engine.lastStep;
  const scrubbingForward = !isFirstUpdate && timeSinceLastUpdate < SCRUB_THRESHOLD_MS;
  const snap = goingBackwards || scrubbingForward;

  engine.lastUpdateTime = now;
  engine.lastStep = step;

  resources.pieces.removeChildren();

  const sources = snap ? null : getAnimationSources(chess);

  for (const row of chess.board()) {
    for (const cell of row) {
      if (!cell) continue;

      const texture = textures[`${cell.color}${cell.type}`];
      if (!texture) continue;

      const sprite = new Sprite({ texture, anchor: 0.5 });
      sprite.scale.set(squareSize / texture.width);

      const target = squareToPixel(cell.square, squareSize, 'white');
      const isAnimating = sources?.get(cell.square);

      if (isAnimating) {
        const start = squareToPixel(isAnimating, squareSize, 'white');
        sprite.position.set(start.x, start.y);
        // Ensure animating pieces render above stationary pieces.
        sprite.zIndex = 1;
        resources.pieces.addChild(sprite);
        engine.animations.add(animate(sprite.position, target, SPRING_CONFIG));
      } else {
        sprite.position.set(target.x, target.y);
        resources.pieces.addChild(sprite);
      }
    }
  }
}
