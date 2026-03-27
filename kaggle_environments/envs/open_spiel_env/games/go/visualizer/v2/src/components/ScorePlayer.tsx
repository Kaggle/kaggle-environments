import { motion } from 'motion/react';
import { useTransition } from '../hooks/useReducedMotion';
import styles from './ScorePlayer.module.css';
import svgSymbolPath from '../assets/icons.svg?url';
import blackStonePath from '../assets/scoreboard-player-black.webp';
import whiteStonePath from '../assets/scoreboard-player-white.webp';

function BrandLogo({ brand }: { brand: string }) {
  return (
    <svg
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      width="128"
      height="128"
      viewBox="0 0 128 128"
      className={styles.brandLogo}
    >
      <use href={`${svgSymbolPath}#${brand}`} />
    </svg>
  );
}

function Pass() {
  return (
    <span className="squiggle-border" aria-hidden="true">
      Pass
    </span>
  );
}

function StoneImage({ color }: { color: 'black' | 'white' }) {
  const imagePath = color === 'black' ? blackStonePath : whiteStonePath;
  return <img src={imagePath} alt={color} role="presentation" className={styles.stoneImage} />;
}

interface Props {
  isActive: boolean;
  isPassed: boolean;
  label: string;
  className?: string;
  color: 'black' | 'white';
  brand: string | null;
}

export function ScorePlayer({ isActive, isPassed, label, brand, color, className }: Props) {
  const transition = useTransition({ type: 'spring', stiffness: 300, damping: 14 });
  const rotate = isActive ? (color === 'white' ? 2 : -2) : 0;

  const classNames = [
    styles.player,
    isActive ? styles.active : undefined,
    'squiggle-border',
    color === 'white' ? styles.isRightAligned : undefined,
    className,
  ].join(' ');

  return (
    <motion.div className={classNames} animate={{ scale: isActive ? 1.05 : 1, rotate }} transition={transition}>
      {isPassed && <Pass />}
      <StoneImage color={color} />
      {brand && <BrandLogo brand={brand} />}
      <span className={styles.playerName}>
        <span className="visually-hidden">{color}</span>
        {label}
      </span>
    </motion.div>
  );
}
