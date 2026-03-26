import { useEffect, useRef } from 'react';
import type { CellValue, GridPos, Territory } from '../types/game.ts';
import { GoPixi } from '../graphics/GoPixi.ts';
import styles from './GoBoard.module.css';

interface Props {
  boardSize: number;
  grid: CellValue[][];
  step: number;
  lastPlayed: GridPos | null;
  atari: GridPos[];
  territory: Territory;
  reducedMotion: boolean;
}

export function GoBoard({ boardSize, grid, step, lastPlayed, atari, territory, reducedMotion }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pixiRef = useRef<GoPixi | null>(null);

  // Init / destroy
  useEffect(() => {
    const pixi = new GoPixi(containerRef.current!, boardSize);
    pixiRef.current = pixi;
    pixi.init();
    return () => {
      pixi.destroy();
      pixiRef.current = null;
    };
  }, [boardSize]);

  // Forward props
  useEffect(() => {
    pixiRef.current?.update({ grid, step, lastPlayed, atari, territory, reducedMotion });
  }, [grid, step, lastPlayed, atari, territory, reducedMotion]);

  return <div ref={containerRef} className={styles.canvas} />;
}
