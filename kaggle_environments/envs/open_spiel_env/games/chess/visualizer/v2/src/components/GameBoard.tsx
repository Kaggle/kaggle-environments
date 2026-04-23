import { memo, useEffect, useRef } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { createGame, type Game } from '../graphics/game';
import useGameStore from '../stores/useGameStore';
import usePreferences, { type PreferencesState } from '../stores/usePreferences';
import styles from './GameBoard.module.css';

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
        game.update(state.game, state.options.step, selectPrefs(usePreferences.getState()));
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
    gameRef.current?.update(chess, step, prefs);
  }, [chess, step, prefs]);

  return (
    <div id="board" className={styles.board}>
      <canvas ref={canvasRef} width={1024} height={1024} />
    </div>
  );
});
