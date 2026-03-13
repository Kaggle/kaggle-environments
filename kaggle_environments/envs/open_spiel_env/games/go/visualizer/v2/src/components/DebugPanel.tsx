// eslint-disable-next-line @typescript-eslint/triple-slash-reference
/// <reference path="../types/popover.d.ts" />
import usePreferences from '../stores/usePreferences';
import useHeroAnimation from '../stores/useHeroAnimation';
import knightRiv from '../assets/kaggle_knight.riv?url';
import queenRiv from '../assets/kaggle_queen.riv?url';
import styles from './DebugPanel.module.css';

const PANEL_ID = 'debug-panel';

const RIVE_FILES = [
  { name: 'kaggle_knight', src: knightRiv },
  { name: 'kaggle_queen', src: queenRiv },
];

export function DebugPanel() {
  const play = useHeroAnimation((s) => s.play);
  const { showTerritory, showAnimations, reducedMotion, toggle } = usePreferences();

  return (
    <>
      <button className={styles.trigger} popovertarget={PANEL_ID} popovertargetaction="toggle">
        Debug
      </button>
      <div id={PANEL_ID} popover="auto" className={styles.panel}>
        <fieldset>
          <legend>Preferences</legend>
          <ul className={styles.list}>
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
        <fieldset>
          <legend>Rive animations</legend>
          <ul className={styles.list}>
            {RIVE_FILES.map(({ name, src }) => (
              <li key={name}>
                <button onClick={() => play(src)}>{name}</button>
              </li>
            ))}
          </ul>
        </fieldset>
      </div>
    </>
  );
}
