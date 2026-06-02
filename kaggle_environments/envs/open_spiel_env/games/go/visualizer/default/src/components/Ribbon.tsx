import { ReactNode } from 'react';
import styles from './Ribbon.module.css';

interface Props {
  children: ReactNode;
}

export function FlagEnd() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 55 116" width="55" height="114">
      <path
        fill="#fff"
        stroke="currentColor"
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
        d="M52 113H3q-2-1-2-3l28-51v-4L1 4q0-2 2-3h49l2 2v108z"
      />
    </svg>
  );
}

export function Ribbon({ children }: Props) {
  return (
    <div className={`grid-pile ${styles.ribbon}`}>
      <FlagEnd />
      <FlagEnd />
      <div className={styles.text}>{children}</div>
    </div>
  );
}
