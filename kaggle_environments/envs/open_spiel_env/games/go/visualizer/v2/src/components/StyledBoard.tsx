import { useRef, useEffect } from 'react';
import { createRenderer } from 'jgoboard';
import useGoStore from '../stores/useGoStore';

export default function StyledBoard() {
  const elem = useRef(null);
  const go = useGoStore((state) => state.go);

  useEffect(() => {
    const options = {
      board: go.board,
      theme: {
        margin: {
          color: 'transparent',
          normal: 0,
          clipped: 0,
        },
        boardShadow: {
          color: 'transparent',
        },
        padding: {
          normal: 20,
          clipped: 10,
        },
        grid: {
          color: 'transparent',
          x: 50,
          y: 50,
        },
        coordinates: {
          top: true,
          right: true,
          bottom: true,
          left: true,
          color: 'transparent',
          font: 'normal 12px sans-serif',
        },
        textures: {
          black: 'images/black.png',
          white: 'images/white.png',
          shadow: 'images/shadow.png',
          board: 'images/board.jpg',
        },
      },
      interactions: { enabled: false },
    };

    const renderer = createRenderer(elem.current, options);

    return () => renderer.destroy();
  }, [go]);

  return <div id="board" ref={elem} />;
}
