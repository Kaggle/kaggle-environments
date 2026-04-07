import { ReactNode, useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { useTransition } from '../hooks/useReducedMotion';
import svgSymbolPath from '../assets/icons.svg?url';
import styles from './WithPopover.module.css';

interface Props {
  children: ReactNode;
  /** This refers to a symbol ID in the svg symbolset (`public/icons.svg`). */
  icon: string;
  id: string;
  label: string;
  onChange?: (open: boolean) => void;
}

export function WithPopover({ children, icon, id, label, onChange }: Props) {
  const [open, setOpen] = useState<boolean | undefined>();
  const transition = useTransition({ duration: 0.2, ease: 'easeOut' });
  const iconPath = `${svgSymbolPath}#${icon}`;
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open !== undefined && onChange) onChange(open);

    if (!open) return;

    // Wait a frame, then focus the popover.
    window.requestAnimationFrame(() => {
      panelRef.current?.focus();
    });

    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (!panelRef.current?.contains(target) && !triggerRef.current?.contains(target)) {
        setOpen(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };

    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  return (
    <div className={styles.wrapper}>
      <button
        ref={triggerRef}
        className={styles.trigger}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((prev) => !prev)}
        data-open={open || undefined}
      >
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <use xlinkHref={iconPath} />
        </svg>
        <span className="visually-hidden">{label}</span>
      </button>
      <motion.div
        ref={panelRef}
        id={id}
        className={styles.panel}
        role="dialog"
        aria-label={label}
        tabIndex={-1}
        aria-hidden={!open || undefined}
        initial={{ opacity: 0, scale: 0.95, display: 'none' }}
        animate={
          open
            ? { display: 'block', opacity: 1, scale: 1 }
            : { opacity: 0, scale: 0.95, transitionEnd: { display: 'none' } }
        }
        transition={transition}
        style={{ transformOrigin: 'left center' }}
      >
        <button
          className={`${styles.closeButton} visually-hidden-unless-focused`}
          onClick={() => {
            setOpen(false);
            triggerRef.current?.focus();
          }}
        >
          Close
        </button>
        <div className={styles.panelInner}>
          <div className={styles.panelInnerInner}>{children}</div>
        </div>
      </motion.div>
    </div>
  );
}
