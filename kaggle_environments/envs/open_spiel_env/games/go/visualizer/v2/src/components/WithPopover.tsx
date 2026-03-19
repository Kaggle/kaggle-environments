import { ReactNode, useEffect, useRef } from 'react';
import styles from './WithPopover.module.css';

interface Props {
  children: ReactNode;
  /** This refers to a symbol ID in the svg symbolset (`public/icons.svg`). */
  icon: string;
  id: string;
  label: string;
}

export function WithPopover({ children, icon, id, label }: Props) {
  const triggerName = `--${id}-trigger`;
  const iconPath = `/icons.svg#${icon}`;
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;

    const handleToggle = (e: Event) => {
      const { newState } = e as ToggleEvent;
      if (newState === 'open') {
        const focusable = panel.querySelector<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        (focusable ?? panel).focus();
      } else {
        triggerRef.current?.focus();
      }
    };

    panel.addEventListener('toggle', handleToggle);
    return () => panel.removeEventListener('toggle', handleToggle);
  }, []);

  return (
    <>
      <button
        ref={triggerRef}
        className={styles.trigger}
        popovertarget={id}
        popovertargetaction="toggle"
        style={{ anchorName: triggerName }}
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <use xlinkHref={iconPath} />
        </svg>
      </button>
      <div
        ref={panelRef}
        id={id}
        popover="auto"
        className={styles.panel}
        role="dialog"
        aria-label={label}
        tabIndex={-1}
      >
        {children}
      </div>
    </>
  );
}
