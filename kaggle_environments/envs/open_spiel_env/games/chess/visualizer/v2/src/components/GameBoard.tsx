import { memo, useEffect, useRef } from 'react';
import { createGame, type Game } from '../graphics/game';
import useGameStore from '../stores/useGameStore';
import styles from './GameBoard.module.css';

export default memo(function GameBoard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameRef = useRef<Game | null>(null);
  const chess = useGameStore((state) => state.game);

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
        game.update(useGameStore.getState().game);
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
    gameRef.current?.update(chess);
  }, [chess]);

  return (
    <div id="board" className={styles.board}>
      <canvas ref={canvasRef} width={1024} height={1024} />
    </div>
  );
});
