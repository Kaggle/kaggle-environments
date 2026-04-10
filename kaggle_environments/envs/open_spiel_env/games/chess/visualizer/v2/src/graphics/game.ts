import type { Chess } from 'chess.js';
import { drawBoard } from './board';
import { engine, initialiseEngine } from './engine';
import { loadPieceTextures, syncPieces } from './pieces';

export interface Game {
  update: (chess: Chess) => void;
  destroy: () => void;
}

export async function createGame(canvas: HTMLCanvasElement): Promise<Game> {
  const eng = engine();
  await initialiseEngine(eng, canvas);

  eng.textures = await loadPieceTextures();

  // Draw board.
  const board = drawBoard(eng.squareSize);
  board.cacheAsTexture(true);
  eng.resources.background.addChild(board);

  return {
    update(chess: Chess) {
      syncPieces(eng, chess);
    },
    destroy() {
      eng.app.destroy({ removeView: true }, { children: true });
    },
  };
}
