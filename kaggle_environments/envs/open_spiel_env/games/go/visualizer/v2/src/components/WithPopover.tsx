import { ReactNode } from 'react';
import styles from './WithPopover.module.css';

interface Props {
  children: ReactNode;
  /** This refers to a symbol ID in the svg symbolset (`public/icons.svg`). */
  icon: string;
  id: string;
}

export function WithPopover({ children, icon, id }: Props) {
  const triggerName = `--${id}-trigger`;
  const iconPath = `/icons.svg#${icon}`;

  return (
    <>
      <button
        className={styles.trigger}
        popovertarget={id}
        popovertargetaction="toggle"
        style={{ anchorName: triggerName }}
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <use xlinkHref={iconPath} />
        </svg>
      </button>
      <div id={id} popover="auto" className={styles.panel}>
        {children}
      </div>
    </>
  );
}
