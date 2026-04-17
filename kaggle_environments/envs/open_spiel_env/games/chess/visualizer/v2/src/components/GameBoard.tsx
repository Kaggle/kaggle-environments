import { memo, useEffect, useRef } from 'react';
import { createGame, type Game } from '../graphics/game';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import styles from './GameBoard.module.css';

export default memo(function GameBoard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameRef = useRef<Game | null>(null);
  const chess = useGameStore((state) => state.game);
  const step = useGameStore((state) => state.options.step);
  const reducedMotion = usePreferences((state) => state.reducedMotion);

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
        game.update(state.game, state.options.step, usePreferences.getState().reducedMotion);
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
    gameRef.current?.update(chess, step, reducedMotion);
  }, [chess, step, reducedMotion]);

  return (
    <div id="board" className={styles.board}>
      <canvas ref={canvasRef} width={1024} height={1024} />
    </div>
  );
});
