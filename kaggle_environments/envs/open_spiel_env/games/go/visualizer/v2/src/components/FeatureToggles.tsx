import usePreferences from '../stores/usePreferences.ts';
import styles from './FeatureToggles.module.css';

export function FeatureToggles() {
  const { toggle, showAnimations, showTerritory, reducedMotion } = usePreferences();
  return (
    <fieldset className={styles.fieldset}>
      <legend>Preferences</legend>
      <ul>
        <li>
          <label>
            Live Territory
            <input
              type="checkbox"
              className={styles.switch}
              checked={showTerritory}
              onChange={() => toggle('showTerritory')}
            />
          </label>
        </li>
        <li>
          <label>
            Popover Animations
            <input
              type="checkbox"
              className={styles.switch}
              checked={showAnimations}
              onChange={() => toggle('showAnimations')}
            />
          </label>
        </li>
        <li>
          <label>
            Reduced Motion
            <input
              type="checkbox"
              className={styles.switch}
              checked={reducedMotion}
              onChange={() => toggle('reducedMotion')}
            />
          </label>
        </li>
      </ul>
    </fieldset>
  );
}
