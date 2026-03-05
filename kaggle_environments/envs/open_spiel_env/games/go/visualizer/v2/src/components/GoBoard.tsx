import { useEffect, useRef } from 'react';
import type { CellValue, Captures } from '../types/game.ts';
import { BOARD_PX, POT_AREA_HEIGHT } from '../graphics/constants.ts';
import { GoPixi } from '../graphics/go-pixi.ts';

interface Props {
  boardSize: number;
  grid: CellValue[][];
  step: number;
  lastPlayed: { row: number; col: number } | null;
  captures: Captures;
}

export function GoBoard({ boardSize, grid, step, lastPlayed, captures }: Props) {
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
    pixiRef.current?.update({ grid, step, lastPlayed, captures });
  }, [grid, step, lastPlayed, captures]);

  return <div ref={containerRef} style={{ width: BOARD_PX, height: BOARD_PX + POT_AREA_HEIGHT }} />;
}
