import { motion } from 'motion/react';
import useGameStore from '../stores/useGameStore.ts';
import { useTransition } from '../hooks/useReducedMotion.ts';
import styles from './Vignette.module.css';

export function Vignette() {
  const game = useGameStore((state) => state.game);
  const isCheck = game.isCheck();
  const transition = useTransition({ type: 'spring', stiffness: 120, damping: 20 });

  return (
    <div className={styles.vignetteOuter}>
      <motion.div className={styles.vignette} animate={{ scale: isCheck ? 1 : 2.5 }} transition={transition} />
    </div>
  );
}
