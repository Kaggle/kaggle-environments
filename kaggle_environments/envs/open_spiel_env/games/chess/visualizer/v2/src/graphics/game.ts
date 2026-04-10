import type { Chess } from 'chess.js';
import { drawBoard } from './board';
import { engine, initialiseEngine } from './engine';
import { loadPieceTextureAtlas, syncPieces } from './pieces';

export interface Game {
  update: (chess: Chess, step: number) => void;
  destroy: () => void;
}

export async function createGame(canvas: HTMLCanvasElement): Promise<Game> {
  const eng = engine();
  await initialiseEngine(eng, canvas);

  eng.textures = await loadPieceTextureAtlas();

  // Draw board.
  const board = drawBoard(eng.squareSize);
  board.cacheAsTexture(true);
  eng.resources.background.addChild(board);

  return {
    update(chess: Chess, step: number) {
      syncPieces(eng, chess, step);
    },
    destroy() {
      // Stop any in-flight spring animations before tearing down the stage,
      // otherwise motion keeps writing to Points on destroyed sprites.
      for (const anim of eng.animations) anim.stop();
      eng.animations.clear();
      eng.app.destroy({ removeView: true }, { children: true });
    },
  };
}
