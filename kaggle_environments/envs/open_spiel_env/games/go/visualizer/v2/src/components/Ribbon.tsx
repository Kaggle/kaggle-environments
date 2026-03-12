import { ReactNode } from 'react';
import styles from './Ribbon.module.css';

interface Props {
  children: ReactNode;
}

export function Ribbon({ children }: Props) {
  return (
    <div className={styles.banner}>
      <RibbonGraphic />
      <div className={styles.text}>{children}</div>
    </div>
  );
}

function RibbonGraphic() {
  return (
    <div className={styles.ribbon}>
      <div className={styles.start}>
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 55 114">
          <path
            fill="#fff"
            stroke="currentColor"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
            d="M52 113H3q-2-1-2-3l28-51v-4L1 4q0-2 2-3h49l2 2v108z"
          />
        </svg>
      </div>
      <div className={styles.middle} />
      <div className={styles.end}>
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 55 114">
          <path
            fill="#fff"
            stroke="currentColor"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
            d="M3 1h49l2 3-28 51v4l28 51-2 3H3l-2-2V3z"
          />
        </svg>
      </div>
    </div>
  );
}
