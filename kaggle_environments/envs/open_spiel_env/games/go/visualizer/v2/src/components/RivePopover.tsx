import { useEffect } from 'react';
import mynerveFont from '../assets/rives/mynerve.ttf?url';
import {
  useRive,
  Layout,
  Fit,
  Alignment,
  useViewModel,
  useViewModelInstance,
  useViewModelInstanceString,
  useViewModelInstanceEnum,
  decodeFont,
  FontAsset,
} from '@rive-app/react-webgl2';
import styles from './RivePopover.module.css';

const layout = new Layout({ fit: Fit.Contain, alignment: Alignment.Center });

interface RivePopoverProps {
  src: string;
  text: string;
  color: string;
  onClose: () => void;
}

export function RivePopover({ src, color, text, onClose }: RivePopoverProps) {
  const { rive, RiveComponent } = useRive({
    src,
    layout,
    stateMachines: 'State Machine 1',
    autoplay: true,
    onStateChange: (e) => {
      if (e.data?.toString() === 'exit') onClose();
    },
    assetLoader: (asset) => {
      if (asset.isFont) {
        fetch(mynerveFont).then(async (res) => {
          const arrayBuffer = await res.arrayBuffer();
          const font = await decodeFont(new Uint8Array(arrayBuffer));
          (asset as FontAsset).setFont(font);
          font.unref();
        });
        return true;
      }
      return false;
    },
  });

  const viewModel = useViewModel(rive, { name: 'ViewModel1' });
  const viewModelInstance = useViewModelInstance(viewModel, { useNew: true, rive });
  const { setValue: setText } = useViewModelInstanceString('text', viewModelInstance);
  const { setValue: setColor } = useViewModelInstanceEnum('color', viewModelInstance);

  useEffect(() => {
    if (!rive || !setText || !setColor || !text || !color) return;
    setText(text);
    setColor(color);
  }, [rive, setText, setColor, text, color]);

  return (
    <div className={`grid-pile ${styles.overlay}`} aria-hidden="true">
      <div className={styles.backdrop} />
      <div className={styles.content}>
        <RiveComponent className={styles.canvas} />
      </div>
    </div>
  );
}
