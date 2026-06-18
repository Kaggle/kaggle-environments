import { memo, useEffect, useRef } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { createGame, type Game } from '../graphics/game';
import useGameStore from '../stores/useGameStore';
import usePreferences, { type PreferencesState } from '../stores/usePreferences';
import type { ChessStep } from '../transformers/chessReplayTypes';
import styles from './GameBoard.module.css';

function isBoardUnchanged(step: ChessStep | undefined): boolean {
  if (!step) return false;
  // Terminal step and forfeit step both leave board state matching the prior
  // step — no move animation should play when scrubbing onto them.
  if (step.isTerminal) return true;
  return step.players.some((p) => p.forfeited);
}

const selectPrefs = (s: PreferencesState): PreferencesState => ({
  showHeroAnimations: s.showHeroAnimations,
  showAnnotations: s.showAnnotations,
  showHighlights: s.showHighlights,
  soundEnabled: s.soundEnabled,
  reducedMotion: s.reducedMotion,
});

export default memo(function GameBoard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameRef = useRef<Game | null>(null);
  const chess = useGameStore((state) => state.game);
  const step = useGameStore((state) => state.options.step);
  const boardUnchanged = useGameStore((state) =>
    isBoardUnchanged(state.options.replay.steps.at(state.options.step) as ChessStep | undefined)
  );
  const prefs = usePreferences(useShallow(selectPrefs));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;

    // Defer init so StrictMode cleanup can cancel before we touch WebGL.
    // Without this, two app.init() calls race on the same canvas and the context is corrupted.
    const frameId = requestAnimationFrame(() => {
      createGame(canvas).then((game) => {
        if (cancelled) {
          game.destroy();
          return;
        }
        gameRef.current = game;
        const state = useGameStore.getState();
        const unchanged = isBoardUnchanged(state.options.replay.steps.at(state.options.step) as ChessStep | undefined);
        game.update(state.game, state.options.step, selectPrefs(usePreferences.getState()), unchanged);
      });
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
      gameRef.current?.destroy();
      gameRef.current = null;
    };
  }, []);

  useEffect(() => {
    gameRef.current?.update(chess, step, prefs, boardUnchanged);
  }, [chess, step, prefs, boardUnchanged]);

  return (
    <div id="board" className={styles.board}>
      <canvas ref={canvasRef} width={1024} height={1024} />
    </div>
  );
});
