import usePreferences from '../stores/usePreferences.ts';
import styles from './FeatureToggles.module.css';

export function FeatureToggles() {
  const { toggle, showHeroAnimations, showTerritory, reducedMotion } = usePreferences();
  return (
    <fieldset className={styles.fieldset}>
      <legend className="visually-hidden">Preferences</legend>
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
            Hero Animations
            <input
              type="checkbox"
              className={styles.switch}
              checked={showHeroAnimations}
              onChange={() => toggle('showHeroAnimations')}
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
