import useGameStore from '../stores/useGameStore';
import { GoBoard } from '../components/go-board';

type CellValue = '.' | 'B' | 'W';

export default function StyledBoard() {
  const game = useGameStore((state) => state.game);
  const state = game.currentState();
  const size = game.boardSize;
  const step = state.moveNumber - 1;

  const grid: CellValue[][] = [];
  for (let y = 0; y < size; y++) {
    const line: CellValue[] = [];
    for (let x = 0; x < size; x++) {
      const value = state.intersectionAt(y, x).value!;
      const cells: { [key: string]: CellValue } = { 'black': 'B', 'white': 'W', 'empty': '.' };
      line.push(cells[value]);
    }
    grid.push(line);
  }

  let lastPlayed = null;
  if (state.playedPoint) {
    lastPlayed = {
      row: state.playedPoint.y,
      col: state.playedPoint.x,
    };
  }

  const captures = {
    black: state.blackStonesCaptured,
    white: state.whiteStonesCaptured,
  };

  return (
    <div id="board">
      <GoBoard boardSize={game.boardSize} grid={grid} step={step} lastPlayed={lastPlayed} captures={captures} />
    </div>
  );
}
