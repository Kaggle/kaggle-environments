import usePreferences from '../stores/usePreferences';
import styles from './FeatureToggles.module.css';

export function FeatureToggles() {
  const { toggle, showHeroAnimations, showAnnotations, showHighlights, reducedMotion } = usePreferences();
  return (
    <fieldset className={styles.fieldset}>
      <legend className="visually-hidden">Preferences</legend>
      <ul>
        <li>
          <label>
            Pop Up Animations
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
        <li>
          <label>
            Move Highlights
            <input
              type="checkbox"
              className={styles.switch}
              checked={showHighlights}
              onChange={() => toggle('showHighlights')}
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
              onChange={() => toggle('showAnnotations')}
            />
          </label>
        </li>
      </ul>
    </fieldset>
  );
}
