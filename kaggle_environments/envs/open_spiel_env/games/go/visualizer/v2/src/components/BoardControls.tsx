import arrowPath from '../assets/arrow.webp';
import { WithPopover } from './WithPopover.tsx';
import { FeatureToggles } from './FeatureToggles.tsx';
import useGameStore from '../stores/useGameStore';
import styles from './BoardControls.module.css';

export default function BoardControls() {
  const game = useGameStore((state) => state.game);

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
