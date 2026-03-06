import { memo } from 'react';
import { CellValue } from '../types/game.ts';
import { GoBoard } from './GoBoard';
import { DebugPanel } from './DebugPanel';
import useGameStore from '../stores/useGameStore';
import { tenukiLogger } from '../utils/tenukiLogger';
import styles from './GameBoard.module.css';

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
  const territory = {
    black: step === 1 ? [] : scorer.black.map((point: { x: number; y: number }) => ({ row: point.y, col: point.x })),
    white: step === 1 ? [] : scorer.white.map((point: { x: number; y: number }) => ({ row: point.y, col: point.x })),
  };

  const atari = state.intersections
    .filter((intersection) => state.inAtari(intersection.y, intersection.x))
    .map((intersection) => ({ row: intersection.y, col: intersection.x }));

  return (
    <div id="board">
      <div className={styles.boardAnchor}>
        <GoBoard
          boardSize={size}
          grid={grid}
          step={step}
          lastPlayed={played}
          captures={captures}
          atari={atari}
          territory={territory}
        />
      </div>
      <DebugPanel />
    </div>
  );
});
