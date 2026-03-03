import { useRef, useEffect } from 'react';
// import { createRenderer } from 'jgoboard';
import useGameStore from '../stores/useGameStore';

export default function StyledBoard() {
  const boardRef = useRef(null);
  const game = useGameStore((state) => state.game);

  useEffect(() => {
    const target = boardRef.current;
    console.log(target);
    // const options = {
    //   board: game.board,
    //   theme: {
    //     margin: {
    //       color: 'transparent',
    //     },
    //     boardShadow: {
    //       color: 'transparent',
    //     },
    //     padding: {
    //       normal: 20,
    //     },
    //     grid: {
    //       color: 'transparent',
    //       x: 50,
    //       y: 50,
    //     },
    //     coordinates: {
    //       top: true,
    //       right: true,
    //       bottom: true,
    //       left: true,
    //       color: 'transparent',
    //       font: 'sans-serif',
    //     },
    //     textures: {
    //       black: 'images/black.png',
    //       white: 'images/white.png',
    //       shadow: 'images/shadow.png',
    //       board: 'images/board.jpg',
    //     },
    //   },
    //   interactions: { enabled: false },
    // };

    // const renderer = createRenderer(target, options);

    // return () => renderer.destroy();
  }, [game]);

  return <div id="board" ref={boardRef} />;
}
