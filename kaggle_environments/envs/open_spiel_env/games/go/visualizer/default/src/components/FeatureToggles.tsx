import usePreferences from '../stores/usePreferences';
import styles from './FeatureToggles.module.css';
import { trackEvent } from '../utils/analytics';

export function FeatureToggles() {
  const { toggle, showHeroAnimations, showTerritory, showAnnotations, reducedMotion } = usePreferences();
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
              onChange={() => {
                toggle('showTerritory');
                trackEvent(`settings-live-territory-${showTerritory ? 'off' : 'on'}`);
              }}
            />
          </label>
        </li>
        <li>
          <label>
            Pop Up Animations
            <input
              type="checkbox"
              className={styles.switch}
              checked={showHeroAnimations}
              onChange={() => {
                toggle('showHeroAnimations');
                trackEvent(`settings-pop-up-animations-${showHeroAnimations ? 'off' : 'on'}`);
              }}
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
              onChange={() => {
                toggle('reducedMotion');
                trackEvent(`settings-reduce-motion-${reducedMotion ? 'off' : 'on'}`);
              }}
            />
          </label>
        </li>
        <li className={styles.annotationsToggle}>
          <label>
            Board Annotations
            <input
              type="checkbox"
              className={styles.switch}
              checked={showAnnotations}
              onChange={() => {
                toggle('showAnnotations');
                trackEvent(`settings-board-annotations-${showAnnotations ? 'off' : 'on'}`);
              }}
            />
          </label>
        </li>
      </ul>
    </fieldset>
  );
}
