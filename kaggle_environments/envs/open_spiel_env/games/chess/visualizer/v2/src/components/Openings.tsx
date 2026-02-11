import { useEffect } from 'react';
import useChessStore from '../stores/useChessStore';

// Openings data from the Lichess Chess Openings project under the CC licence.
// https://github.com/lichess-org/chess-openings/blob/master/COPYING.txt
import openings from '../data/openings.json';

export default function Openings() {
  const chess = useChessStore((state) => state.chess);

  useEffect(() => {
    const opening = openings.find((opening) => chess.fen().includes(opening.fen));

    if (opening) console.log(`*** ${opening.name} ***`);
  }, [chess]);

  return null;
}
