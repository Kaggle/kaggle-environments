import { WithPopover } from './WithPopover.tsx';
import { FeatureToggles } from './FeatureToggles.tsx';
import useGameStore from '../stores/useGameStore';
import styles from './BoardControls.module.css';
import arrowPath from '../assets/arrow.webp';

export function BoardControls() {
  const game = useGameStore((s) => s.game);
  const showCta = game.currentState().moveNumber === 0;

  return (
    <div className={styles.boardControls}>
      <WithPopover id="info" icon="info" label="Game info">
        <p>
          Go is an ancient, two-player game in which players try to control more territory on a grid by strategically
          placing black and white stones. This game is based on Tromp-Taylor rules.
        </p>
      </WithPopover>
      <WithPopover id="settings" icon="settings" label="Settings">
        <FeatureToggles />
      </WithPopover>

      {showCta && (
        <div className={styles.settingsCta}>
          <img src={arrowPath} width="57" alt="137" aria-hidden="true" />
          <p>Customise your experience</p>
        </div>
      )}
    </div>
  );
}
