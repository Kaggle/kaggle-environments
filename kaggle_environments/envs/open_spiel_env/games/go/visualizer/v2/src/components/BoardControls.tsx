import { WithPopover } from './WithPopover.tsx';
import { FeatureToggles } from './FeatureToggles.tsx';
import styles from './BoardControls.module.css';

export function BoardControls() {
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
    </div>
  );
}
