import { useRive, Layout, Fit, Alignment, useViewModel, useViewModelInstance, useViewModelInstanceString } from '@rive-app/react-webgl2';
import styles from './RivePopover.module.css';

const layout = new Layout({ fit: Fit.Contain, alignment: Alignment.Center });

interface RivePopoverProps {
  src: string;
  text: string;
  onClose: () => void;
}

export function RivePopover({ src, text, onClose }: RivePopoverProps) {
  const { rive, RiveComponent } = useRive({
    src,
    autoplay: true,
    autoBind: false,
    layout,
    onStop: () => onClose(),
  });

  const viewModel = useViewModel(rive, { name: 'ViewModel1' });
  const viewModelInstance = useViewModelInstance(viewModel, { rive });
  const { setValue } = useViewModelInstanceString('text', viewModelInstance);
  
  setValue(text);

  return (
    <div className={`grid-pile ${styles.overlay}`} aria-hidden="true">
      <div className={styles.backdrop} />
      <div className={styles.content}>
        <RiveComponent className={styles.canvas} />
      </div>
    </div>
  );
}
