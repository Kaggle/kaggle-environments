import { useRive, Layout, Fit, Alignment } from '@rive-app/react-webgl2';
import styles from './RivePopover.module.css';

const layout = new Layout({ fit: Fit.Contain, alignment: Alignment.Center });

interface RivePopoverProps {
  src: string;
  onClose: () => void;
}

export function RivePopover({ src, onClose }: RivePopoverProps) {
  const { RiveComponent } = useRive({
    src,
    autoplay: true,
    layout,
    onStop: () => onClose(),
  });

  return (
    <div className={`grid-pile ${styles.overlay}`} aria-hidden="true">
      <div className={styles.backdrop} />
      <div className={styles.content}>
        <RiveComponent className={styles.canvas} />
      </div>
    </div>
  );
}
