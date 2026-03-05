import { memo } from 'react';
import { CellValue } from '../types/game.ts';
import { GoBoard } from './GoBoard';
import useGameStore from '../stores/useGameStore';
import { tenukiLogger } from '../utils/tenukiLogger';

export default memo(function GameBoard() {
  const game = useGameStore((state) => state.game);

  tenukiLogger(game);

  const state = game.currentState();
  const size = game.boardSize;
  const step = state.moveNumber;

  const grid: CellValue[][] = Array.from({ length: size }, () => new Array(size));
  state.intersections.forEach((item) => {
    const cells: { [key: string]: CellValue } = { 'black': 'B', 'white': 'W', 'empty': '.' };
    grid[item.y][item.x] = cells[item.value!];
  });

  let played = null;
  if (state.playedPoint) {
    played = {
      row: state.playedPoint.y,
      col: state.playedPoint.x,
    };
  }

  const captures = {
    black: state.blackStonesCaptured,
    white: state.whiteStonesCaptured,
  };

  const scorer = game._scorer.territory(game);

  // @ts-expect-error no unused vars
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const territory = {
    black: scorer.black.map((point: { x: number; y: number }) => ({ row: point.y, col: point.x })),
    white: scorer.white.map((point: { x: number; y: number }) => ({ row: point.y, col: point.x })),
  };

  const atari = state.intersections
    .filter((intersection) => state.inAtari(intersection.y, intersection.x))
    .map((intersection) => ({ row: intersection.y, col: intersection.x }));

  return (
    <div id="board">
      <GoBoard boardSize={size} grid={grid} step={step} lastPlayed={played} captures={captures} atari={atari} />
    </div>
  );
});
