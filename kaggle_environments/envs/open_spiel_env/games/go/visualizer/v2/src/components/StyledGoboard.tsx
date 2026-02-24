import { useRef, useEffect } from 'react';
import { createRenderer } from 'jgoboard';
import useGoStore from '../stores/useGoStore';

export default function StyledGoboard() {
  const elem = useRef(null);
  const go = useGoStore((state) => state.go);

  useEffect(() => {
    const options = {
      board: go.board,
      theme: 'kaya-large',
      interactions: { enabled: false },
    };

    const renderer = createRenderer(elem.current, options);

    return () => renderer.destroy();
  }, [go]);

  return <div ref={elem}></div>;
}
