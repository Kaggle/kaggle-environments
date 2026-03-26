import { useRef } from 'react';
import potImg from '../assets/pot.webp';
import styles from './ScorePanel.module.css';

interface ScorePotProps {
  className?: string;
  count: number;
  stoneImg: string;
  label: string;
}

const SCATTER_RADIUS = 60;

interface ScatterPos {
  x: number;
  y: number;
}

function generateScatterPos(): ScatterPos {
  const angle = Math.random() * Math.PI * 2;
  const r = SCATTER_RADIUS * Math.sqrt(Math.random());
  return { x: Math.cos(angle) * r, y: Math.sin(angle) * r };
}

export default function ScorePot({ count, stoneImg, label, className }: ScorePotProps) {
  const positionsRef = useRef<Map<number, ScatterPos>>(new Map());

  return (
    <div className={`grid-pile ${styles.potArea} ${className}`} role="img" aria-label={label}>
      <img src={potImg} className={styles.potImage} alt="" aria-hidden="true" draggable="false" />
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
