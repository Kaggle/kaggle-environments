import { memo, useRef } from 'react';
import blackImg from '../assets/stone-black.png';
import whiteImg from '../assets/stone-white.png';

import potImg from '../graphics/sprites/originals/pot.png';
import useGameStore from '../stores/useGameStore';
import styles from './CapturePots.module.css';

const SCATTER_RADIUS = 75;

interface ScatterPos {
  x: number;
  y: number;
}

function generateScatterPos(): ScatterPos {
  const angle = Math.random() * Math.PI * 2;
  const r = SCATTER_RADIUS * Math.sqrt(Math.random());
  return { x: Math.cos(angle) * r, y: Math.sin(angle) * r };
}

function Pot({ count, stoneImg }: { count: number; stoneImg: string }) {
  const positionsRef = useRef<Map<number, ScatterPos>>(new Map());

  return (
    <div className={styles.potArea}>
      <img src={potImg} className={styles.potImage} alt="" draggable={false} />
      {/* eslint-disable-next-line react-hooks/refs */}
      {Array.from({ length: count }, (_, i) => {
        if (!positionsRef.current.has(i)) {
          positionsRef.current.set(i, generateScatterPos());
        }
        const pos = positionsRef.current.get(i)!;

        return (
          <img
            key={i}
            src={stoneImg}
            className={styles.prisoner}
            style={{ translate: `${pos.x}% ${pos.y}%` }}
            draggable={false}
            alt=""
          />
        );
      })}
    </div>
  );
}

export default memo(function CapturePots() {
  const game = useGameStore((state) => state.game);
  const state = game.currentState();

  return (
    <section className={styles.pots} aria-label="Capture pots">
      <Pot count={state.whiteStonesCaptured} stoneImg={whiteImg} />
      <Pot count={state.blackStonesCaptured} stoneImg={blackImg} />
    </section>
  );
});
