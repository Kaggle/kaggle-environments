import type { Chess } from 'chess.js';
import { drawBoard } from './board';
import { engine, initialiseEngine } from './engine';
import { createToHighlightTexture, syncHighlights } from './highlights';
import { loadPieceTextureAtlas, syncPieces } from './pieces';

export interface Game {
  update: (chess: Chess, step: number) => void;
  destroy: () => void;
}

export async function createGame(canvas: HTMLCanvasElement): Promise<Game> {
  const eng = engine();
  await initialiseEngine(eng, canvas);

  eng.textures = await loadPieceTextureAtlas();
  eng.toHighlightTexture = createToHighlightTexture(eng.app.renderer, eng.squareSize);

  // Draw board, and cache so we only draw it once.
  const board = drawBoard(eng.squareSize, eng.boardOffset, eng.textures['dark-tile.png']);
  board.cacheAsTexture(true);
  eng.resources.background.addChild(board);

  return {
    update(chess: Chess, step: number) {
      // Stop all in-flight animations before rebuilding sprites.
      for (const anim of eng.animations) anim.stop();
      eng.animations.clear();

      syncHighlights(eng, chess);
      syncPieces(eng, chess, step);
    },
    destroy() {
      // TODO(pim-at-stink): https://github.com/pixijs/pixijs/issues/11977
      board.cacheAsTexture(false);
      // Stop any in-flight spring animations before tearing down the stage,
      // otherwise motion keeps writing to Points on destroyed sprites.
      for (const anim of eng.animations) anim.stop();
      eng.animations.clear();
      eng.app.destroy({ removeView: true }, { children: true });
    },
  };
}
