import type { Chess } from 'chess.js';
import { drawBoard } from './board';
import { engine, initialiseEngine, type Engine } from './engine';
import { syncHighlights } from './highlights';
import { loadPieceTextureAtlas, syncPieces } from './pieces';
import { createTrails } from './trails';
import { SCRUB_THRESHOLD_MS } from '../constants';
import type { PreferencesState } from '../stores/usePreferences';

export interface Game {
  update: (chess: Chess, step: number, prefs: PreferencesState) => void;
  destroy: () => void;
}

function detectSnap(eng: Engine, step: number, reducedMotion: boolean, now: number): boolean {
  const isFirstUpdate = eng.lastUpdateTime === 0;
  const timeSinceLastUpdate = now - eng.lastUpdateTime;
  const goingBackwards = step < eng.lastStep;
  const scrubbingForward = !isFirstUpdate && timeSinceLastUpdate < SCRUB_THRESHOLD_MS;
  return reducedMotion || goingBackwards || scrubbingForward;
}

export async function createGame(canvas: HTMLCanvasElement): Promise<Game> {
  const eng = engine();
  await initialiseEngine(eng, canvas);

  eng.textures = await loadPieceTextureAtlas();

  // Draw board, and cache so we only draw it once.
  const board = drawBoard(eng.squareSize, eng.boardOffset, eng.textures['dark-tile.png']);
  board.cacheAsTexture(true);
  eng.resources.background.addChild(board);

  const trails = createTrails(eng);

  return {
    update(chess: Chess, step: number, prefs: PreferencesState) {
      // Stop all in-flight animations before rebuilding sprites.
      for (const anim of eng.animations) anim.stop();
      eng.animations.clear();

      const now = performance.now();
      const snap = detectSnap(eng, step, prefs.reducedMotion, now);
      eng.lastUpdateTime = now;
      eng.lastStep = step;

      trails.clear();

      syncHighlights(eng, chess, prefs);
      syncPieces(eng, chess, snap);
    },
    destroy() {
      trails.destroy();
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
