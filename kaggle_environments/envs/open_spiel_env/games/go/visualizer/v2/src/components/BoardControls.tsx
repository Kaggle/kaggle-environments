import { useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import arrowPath from '../assets/arrow.webp';
import svgSymbolPath from '../assets/icons.svg?url';
import usePreferences from '../stores/usePreferences';
import useGameStore from '../stores/useGameStore';
import { useTransition } from '../hooks/useReducedMotion';
import popoverStyles from './WithPopover.module.css';
import { WithPopover } from './WithPopover.tsx';
import { FeatureToggles } from './FeatureToggles';
import styles from './BoardControls.module.css';

export default function BoardControls() {
  const game = useGameStore((state) => state.game);
  const { toggle, soundEnabled } = usePreferences();
  const transition = useTransition({ duration: 0.3 });

  const iconPath = useMemo(() => {
    const soundIcon = soundEnabled ? 'sound-on' : 'sound-off';
    return `${svgSymbolPath}#${soundIcon}`;
  }, [soundEnabled]);

  // React 18 doesn't support the `inert` HTML attribute as a prop, so we
  // set it imperatively via a ref callback. This can be replaced with a
  // regular `inert` prop once the project upgrades to React 19+.
  const inertRef = (el: HTMLElement | null) => {
    if (!el) return;
    if (game.gameOver) el.setAttribute('inert', '');
    else el.removeAttribute('inert');
  };

  return (
    <div className={styles.boardControls} ref={inertRef}>
      <WithPopover id="info" icon="info" label="Game info">
        <p>
          Go is an ancient, two-player game in which players try to control more territory on a grid by strategically
          placing black and white stones. This game follows Tromp-Taylor rules.
        </p>
      </WithPopover>
      <label className={`${popoverStyles.trigger} ${styles.soundToggle}`}>
        <input
          type="checkbox"
          checked={soundEnabled}
          onChange={() => toggle('soundEnabled')}
          className="visually-hidden"
        />
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <use xlinkHref={iconPath} />
        </svg>
        <span className="visually-hidden">Sound</span>
      </label>
      <WithPopover id="settings" icon="settings" label="Settings">
        <FeatureToggles />
      </WithPopover>

      <AnimatePresence>
        {game.gameStart && (
          <motion.div
            className={styles.settingsCta}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={transition}
          >
            <img src={arrowPath} width="57" alt="137" aria-hidden="true" />
            <p>Customise your experience</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
