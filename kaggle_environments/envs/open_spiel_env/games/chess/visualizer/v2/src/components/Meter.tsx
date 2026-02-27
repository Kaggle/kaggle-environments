import { useEffect, useState, useRef } from 'react';
import useChessStore from '../stores/useChessStore';

export default function Meter() {
  const workerRef = useRef<Worker | null>(null);
  const chess = useChessStore((state) => state.chess);
  const [percent, setPercent] = useState(0);

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
          setPercent(percent);
        }
      };

      workerRef.current.postMessage('stop');
      workerRef.current.postMessage(`position fen ${chess.fen()}`);
      workerRef.current.postMessage('go depth 1');
    }
  }, [chess, setPercent]);

  return <meter value={percent} min="10" max="90" />;
}
