import { useMemo } from 'react';
import useGameStore from '../stores/useGameStore';
import { GoBoard } from '../components/go-board';

type CellValue = '.' | 'B' | 'W';

export default function StyledBoard() {
  const game = useGameStore((state) => state.game);

  const { size, step, grid, played, captures } = useMemo(() => {
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

    return { size, step, grid, played, captures };
  }, [game]);

  return (
    <div id="board">
      <GoBoard boardSize={size} grid={grid} step={step} lastPlayed={played} captures={captures} />
    </div>
  );
}
