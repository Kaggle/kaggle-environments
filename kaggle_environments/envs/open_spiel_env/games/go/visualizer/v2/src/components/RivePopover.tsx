import { useRive, Layout, Fit, Alignment } from '@rive-app/react-webgl2';
import styles from './RivePopover.module.css';

interface RivePopoverProps {
  buffer: ArrayBuffer;
  onClose: () => void;
}

export function RivePopover({ buffer, onClose }: RivePopoverProps) {
  const { RiveComponent } = useRive({
    buffer,
    autoplay: true,
    layout: new Layout({ fit: Fit.Contain, alignment: Alignment.Center }),
    onStop: () => onClose(),
  });

  return (
    <div className={`grid-pile ${styles.overlay}`}>
      <div className={styles.backdrop} />
      <div className={styles.content}>
        <RiveComponent className={styles.canvas} />
      </div>
    </div>
  );
}
