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
    const atari = state.intersections.filter((intersection) => state.inAtari(intersection.x, intersection.y));
    console.log('atari', atari, state);
  }, [game]);

  return <div id="board" ref={boardRef} />;
}
