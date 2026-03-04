import { useEffect } from 'react';
import useGameStore from '../stores/useGameStore';
import { GoBoard } from '../components/go-board';
import { useState } from 'react';

type CellValue = '.' | 'B' | 'W';

export default function StyledBoard() {
  const game = useGameStore((state) => state.game);

  const [size, setSize] = useState(0);
  const [step, setStep] = useState(0);
  const [grid, setGrid] = useState<CellValue[][]>([]);
  const [played, setPlayed] = useState<{row: number, col: number} | null>(null);
  const [captures, setCaptures] = useState({black: 0, white: 0});

  useEffect(() => {
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

    setSize(size);
    setStep(step);
    setGrid(grid);
    setPlayed(played);
    setCaptures(captures);
  }, [game, setSize, setStep, setGrid, setPlayed, setCaptures]);
  
  return (
    <div id="board">
      <GoBoard boardSize={size} grid={grid} step={step} lastPlayed={played} captures={captures} />
    </div>
  );
}
