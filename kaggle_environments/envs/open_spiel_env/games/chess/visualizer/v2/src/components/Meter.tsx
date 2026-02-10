import { useEffect, useRef } from 'react';
import useChessStore from '../stores/useChessStore';

const Meter = () => {
  const workerRef = useRef<Worker>(null);
  const chess = useChessStore((state) => state.chess);

  useEffect(() => {
    workerRef.current = new Worker('./scripts/stockfish-17.1-lite-single-03e3232.js');

    return () => {
      if (workerRef.current) workerRef.current.terminate();
    };
  }, []);

  useEffect(() => {
    if (workerRef.current) {
      workerRef.current.onmessage = (event: MessageEvent) => {
        const match = event.data.match(/score cp (-?\d+)/);
        if (match) {
          let percent = Math.round(50 + 50 * (2 / (1 + Math.exp(-0.00368208 * Number(match.at(1)))) - 1));
          if (chess.turn() == 'b') percent = 100 - percent;
          console.log(`score w ${percent}% / b ${100 - percent}%`);
        }
      };

      workerRef.current.postMessage('stop');
      workerRef.current.postMessage(`position fen ${chess.fen()}`);
      workerRef.current.postMessage('go depth 1');
    }
  }, [chess]);

  return null;
};

export default Meter;
