// eslint-disable-next-line @typescript-eslint/triple-slash-reference
/// <reference path="../types/popover.d.ts" />
import { useState, useCallback } from 'react';
import { RivePopover } from './RivePopover';
import { useRiveFiles } from '../hooks/useRiveFiles';
import usePreferences from '../stores/usePreferences';
import styles from './DebugPanel.module.css';

const PANEL_ID = 'debug-panel';

export function DebugPanel() {
  const riveFiles = useRiveFiles();
  const [active, setActive] = useState<string | null>(null);
  const close = useCallback(() => setActive(null), []);

  const { showTerritory, showAnimations, reducedMotion, toggle } = usePreferences();

  const activeEntry = riveFiles.find((e) => e.file === active);

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
            {riveFiles.map(({ name, file, buffer }) => (
              <li key={file}>
                <button onClick={() => setActive(file)} data-active={active === file || undefined} disabled={!buffer}>
                  {name}
                </button>
              </li>
            ))}
          </ul>
        </fieldset>
      </div>
      {showAnimations && activeEntry?.buffer && (
        <RivePopover key={active} buffer={activeEntry.buffer} onClose={close} />
      )}
    </>
  );
}
