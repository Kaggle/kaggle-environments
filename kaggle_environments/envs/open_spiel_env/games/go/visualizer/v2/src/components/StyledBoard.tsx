import { useRef, useEffect } from 'react';
import useGameStore from '../stores/useGameStore';
import { tenukiToString } from '../utils/tenukiToString';

export default function StyledBoard() {
  const boardRef = useRef(null);
  const game = useGameStore((state) => state.game);

  useEffect(() => {
    console.log(tenukiToString(game));

    // Check for stones in atari
    const state = game.currentState();
    const atari = state.intersections.filter((intersection) => state.inAtari(intersection.y, intersection.x));
    atari.forEach(item => {
      console.log('atari', game.coordinatesFor(item.y, item.x));
    });

    // Check for territories
    const territories = game._scorer.territory(game);

    territories.white.forEach((item: {x: number, y: number}) => {
      console.log('territory', 'W', game.coordinatesFor(item.y, item.x));
    })

    territories.black.forEach((item: {x: number, y: number}) => {
      console.log('territory', 'B', game.coordinatesFor(item.y, item.x));
    })
    
  }, [game]);

  return <div id="board" ref={boardRef} />;
}
