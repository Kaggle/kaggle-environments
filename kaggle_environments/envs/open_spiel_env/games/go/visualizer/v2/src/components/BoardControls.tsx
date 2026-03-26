import { useMemo } from 'react';
import arrowPath from '../assets/arrow.webp';
import svgSymbolPath from '../assets/icons.svg?url';
import usePreferences from '../stores/usePreferences.ts';
import useGameStore from '../stores/useGameStore';
import popoverStyles from './WithPopover.module.css';
import { WithPopover } from './WithPopover.tsx';
import { FeatureToggles } from './FeatureToggles.tsx';
import styles from './BoardControls.module.css';

export default function BoardControls() {
  const game = useGameStore((state) => state.game);
  const { toggle, soundEnabled } = usePreferences();

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
          placing black and white stones. This game is based on Tromp-Taylor rules.
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

      {game.gameStart && (
        <div className={styles.settingsCta}>
          <img src={arrowPath} width="345" alt="368" aria-hidden="true" />
          <p>Customise your experience</p>
        </div>
      )}
    </div>
  );
}
