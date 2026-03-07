// eslint-disable-next-line @typescript-eslint/triple-slash-reference
/// <reference path="../types/popover.d.ts" />
import { useEffect, useRef } from 'react';
import { useRive, Layout, Fit, Alignment } from '@rive-app/react-webgl2';
import styles from './RivePopover.module.css';

interface RivePopoverProps {
  buffer: ArrayBuffer;
  onClose: () => void;
}

export function RivePopover({ buffer, onClose }: RivePopoverProps) {
  const popoverRef = useRef<HTMLDivElement>(null);

  const { RiveComponent } = useRive({
    buffer,
    autoplay: true,
    layout: new Layout({ fit: Fit.Contain, alignment: Alignment.Center }),
    onStop: () => onClose(),
  });

  // Show popover on mount
  useEffect(() => {
    popoverRef.current?.showPopover();
  }, []);

  // Handle light-dismiss and Escape
  useEffect(() => {
    const el = popoverRef.current;
    if (!el) return;

    const handleToggle = (e: ToggleEvent) => {
      if (e.newState === 'closed') onClose();
    };
    el.addEventListener('toggle', handleToggle);
    return () => el.removeEventListener('toggle', handleToggle);
  }, [onClose]);

  return (
    <div ref={popoverRef} popover="auto" className={styles.popover}>
      <div className={styles.content}>
        <button onClick={onClose} className={styles.closeButton}>
          &times;
        </button>
        <RiveComponent className={styles.canvas} />
      </div>
    </div>
  );
}
