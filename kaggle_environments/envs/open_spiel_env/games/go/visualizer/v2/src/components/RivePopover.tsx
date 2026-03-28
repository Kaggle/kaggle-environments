import { useCallback } from 'react';
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
  const transition = useTransition({ duration: 0.35 });
  const overlayRef = useCallback((el: HTMLDivElement | null) => {
    if (el && !el.matches(':popover-open')) el.showPopover();
  }, []);
  const { RiveComponent } = useRive({
    src,
    layout,
    stateMachines: 'State Machine 1',
    autoplay: true,
    autoBind: true,
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
    onRiveReady: (r) => {
      const textString = r.viewModelInstance?.string('text');
      if (textString) textString.value = text;
      const colorEnum = r.viewModelInstance?.enum('color');
      if (colorEnum) colorEnum.value = color;
    },
  });

  return (
    <motion.div
      ref={overlayRef}
      popover="manual"
      className={`grid-pile ${styles.overlay}`}
      aria-hidden="true"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
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
