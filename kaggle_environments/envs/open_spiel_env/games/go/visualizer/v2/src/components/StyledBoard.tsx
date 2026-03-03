import { useEffect } from 'react';
import useGameStore from '../stores/useGameStore';
import { tenukiToString } from '../utils/tenukiToString';
import { GoBoard } from '../components/go-board';

type CellValue = '.' | 'B' | 'W';

export default function StyledBoard() {
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

  const state = game.currentState();
  const size = game.boardSize;
  const step = options?.step!;

  const grid: CellValue[][] = [];
  for (let y = 0; y < size; y++) {
    const line: CellValue[] = [];
    for (let x = 0; x < size; x++) {
      const value = state.intersectionAt(y, x).value;
      if (value === 'black') line.push('B');
      if (value === 'white') line.push('W');
      if (value === 'empty') line.push('.');
    }
    grid.push(line);
  }

  let lastPlayed = null;
  if (state.playedPoint) {
    lastPlayed = {
      row: state.playedPoint.y,
      col: state.playedPoint.x,
    }
  }

  const captures = { 
    black: state.blackStonesCaptured, 
    white: state.whiteStonesCaptured,
  }

  useEffect(() => {
    console.log(tenukiToString(game));

    // // Check for stones in atari
    // const state = game.currentState();
    // const atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
    // atari.forEach(item => {
    //   console.log('atari', game.coordinatesFor(item.y, item.x));
    // });

    // // Check for territories
    // const territories = game._scorer.territory(game);

    // territories.white.forEach((item: {x: number, y: number}) => {
    //   console.log('territory', 'W', game.coordinatesFor(item.y, item.x));
    // })

    // territories.black.forEach((item: {x: number, y: number}) => {
    //   console.log('territory', 'B', game.coordinatesFor(item.y, item.x));
    // })
    
  }, [game]);

  return (
    <div id="board">
      <GoBoard boardSize={game.boardSize} grid={grid} step={step} lastPlayed={lastPlayed} captures={captures}/>
    </div>
  );
}
