import { useEffect } from 'react';
import useGameStore from '../stores/useGameStore';

// Openings data from the Lichess Chess Openings project under the CC licence.
// https://github.com/lichess-org/chess-openings/blob/master/COPYING.txt
import openings from '../data/openings.json';

export default function Openings() {
  const game = useGameStore((state) => state.game);

  useEffect(() => {
    const opening = openings.find((opening) => game.fen().includes(opening.fen));

    if (opening) console.log('opening', opening.name);
  }, [game]);

  return null;
}
