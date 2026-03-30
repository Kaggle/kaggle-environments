import { useState } from 'react';
import { motion } from 'motion/react';
import { useTransition } from '../hooks/useReducedMotion';
import mynerveFont from '../assets/rives/mynerve.ttf?url';
import { useRive, Layout, Fit, Alignment, decodeFont, FontAsset } from '@rive-app/react-webgl2';
import styles from './RivePopover.module.css';

const layout = new Layout({ fit: Fit.Contain, alignment: Alignment.Center });

interface RivePopoverProps {
  src: string;
  text: string;
  color: string;
  onClose: () => void;
}

export function RivePopover({ src, color, text, onClose }: RivePopoverProps) {
  const [hold, setHold] = useState(true);
  const transition = useTransition({ duration: 0.35 });
  const { RiveComponent } = useRive({
    src,
    layout,
    stateMachines: 'State Machine 1',
    autoplay: true,
    autoBind: true,
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
    onRiveReady: (rive) => {
      const textString = rive.viewModelInstance?.string('text');
      if (textString) textString.value = text;
      const colorEnum = rive.viewModelInstance?.enum('color');
      if (colorEnum) colorEnum.value = color;
    },
    onStateChange: (event) => {
      const state = event.data?.toString();
      if (state === 'Black' || state === 'White') setHold(false);
      if (state === 'exit') onClose();
    },
  });

  return (
    <motion.div
      className={`grid-pile ${styles.overlay}`}
      aria-hidden="true"
      initial={{ opacity: 0 }}
      animate={{ opacity: hold ? 0 : 1 }}
      exit={{ opacity: 0 }}
      transition={transition}
    >
      <div className={styles.backdrop} />
      <div className={styles.content}>
        <RiveComponent className={styles.canvas} />
      </div>
    </motion.div>
  );
}
