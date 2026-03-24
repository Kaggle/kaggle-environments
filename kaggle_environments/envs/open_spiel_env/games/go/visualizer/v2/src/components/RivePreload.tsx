import passRiv from '../assets/pass.riv?url';
import doublePassRiv from '../assets/double-pass.riv?url';
import firstCaptureRiv from '../assets/first-capture.riv?url';
import criticalHitRiv from '../assets/critical-hit.riv?url';
import dragonLossRiv from '../assets/dragon-loss.riv?url';

export default function RivePreload() {
  return (
    <>
      <link rel="preload" href={passRiv} as="fetch" crossOrigin="anonymous" />
      <link rel="preload" href={doublePassRiv} as="fetch" crossOrigin="anonymous" />
      <link rel="preload" href={firstCaptureRiv} as="fetch" crossOrigin="anonymous" />
      <link rel="preload" href={criticalHitRiv} as="fetch" crossOrigin="anonymous" />
      <link rel="preload" href={dragonLossRiv} as="fetch" crossOrigin="anonymous" />
    </>
  );
}
