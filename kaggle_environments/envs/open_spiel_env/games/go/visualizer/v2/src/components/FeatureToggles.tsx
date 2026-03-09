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
            <input type="checkbox" checked={showTerritory} onChange={() => toggle('showTerritory')} />
            Live Territory
          </label>
        </li>
        <li>
          <label>
            <input type="checkbox" checked={showAnimations} onChange={() => toggle('showAnimations')} />
            Popover Animations
          </label>
        </li>
        <li>
          <label>
            <input type="checkbox" checked={reducedMotion} onChange={() => toggle('reducedMotion')} />
            Reduced Motion
          </label>
        </li>
      </ul>
    </fieldset>
  );
}
