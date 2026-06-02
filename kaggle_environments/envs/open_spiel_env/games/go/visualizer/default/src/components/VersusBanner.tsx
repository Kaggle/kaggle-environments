import { AnimatePresence, motion } from 'motion/react';
import { Ribbon } from './Ribbon';
import useGameStore from '../stores/useGameStore';
import usePreloader from '../stores/usePreloader';
import { useTransition } from '../hooks/useReducedMotion';
import { trackEvent } from '../utils/analytics';
import styles from './VersusBanner.module.css';

export default function VersusBanner() {
  const game = useGameStore((state) => state.game);
  const loaded = usePreloader((s) => s.pixiReady && s.assetsReady);
  const enterTransition = useTransition({ duration: 1.3, ease: [0.22, 1.15, 0.36, 1] });
  const exitTransition = useTransition({ duration: 0.3 });

  if (loaded && game.gameStart) trackEvent('versus-banner');

  return (
    <AnimatePresence>
      {loaded && game.gameStart && (
        <motion.div
          className={styles.versusBanner}
          aria-hidden="true"
          initial={{ y: '-100vh', rotate: 12, opacity: 1 }}
          animate={{ y: '-50%', rotate: 0, opacity: 1, transition: enterTransition }}
          exit={{ opacity: 0, scale: 0.9, transition: exitTransition }}
        >
          <Ribbon>
            {game.blackName ?? 'Black'} vs. {game.whiteName ?? 'White'}
          </Ribbon>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
