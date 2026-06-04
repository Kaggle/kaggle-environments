import { motion } from 'motion/react';
import { useMemo } from 'react';
import blackMarble from '../assets/marbles/black_marble.png';
import blueMarble from '../assets/marbles/blue_marble.png';
import greenMarble from '../assets/marbles/green_marble.png';
import whiteMarble from '../assets/marbles/white_marble.png';
import yellowMarble from '../assets/marbles/yellow_marble.png';

const MARBLES = [blackMarble, blueMarble, greenMarble, whiteMarble, yellowMarble];

// Deterministic pseudo-random based on seed so the same stone in the same pit
// keeps its color across re-renders.
function pickMarble(seed: number) {
  const idx = Math.abs(Math.floor(Math.sin(seed) * 10000)) % MARBLES.length;
  return MARBLES[idx];
}

function pickRotation(seed: number) {
  return Math.abs(Math.sin(seed * 7.31) * 10000) % 360;
}

export default function Stone({ seed }: { seed: number }) {
  const marble = useMemo(() => pickMarble(seed), [seed]);
  const rotation = useMemo(() => pickRotation(seed), [seed]);
  return (
    <motion.img
      layout
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0, opacity: 0 }}
      src={marble}
      alt=""
      className="mancala-stone"
      style={{ transform: `rotate(${rotation}deg)` }}
    />
  );
}
