import { useRef, useEffect } from 'react';
import useGameStore from '../stores/useGameStore';
import { tenukiToString } from '../utils/tenukiToString';

export default function StyledBoard() {
  const boardRef = useRef(null);
  const game = useGameStore((state) => state.game);

  useEffect(() => {
    console.log(tenukiToString(game));
  }, [game]);

  return <div id="board" ref={boardRef} />;
}
