import { memo, useEffect } from 'react';
import HiddenHeader from './HiddenHeader';
import SvgSprite from './SvgSprite';
import BoardControls from './BoardControls';
import GameBoard from './GameBoard';
import Annotation from './Annotation';
import { SoundEffects } from './SoundEffects.tsx';
import VersusBanner from './VersusBanner';
import CheckBanner from './CheckBanner';
import GameOver from './GameOver';
import HeroAnimation from './HeroAnimation';
import useBoardRect from '../hooks/useBoardRect';
import usePreloader from '../stores/usePreloader';
import { assetsReady } from '../utils/preloadAssets';
import styles from './Layout.module.css';
import { Vignette } from './Vignette.tsx';
import PlayerBar from './PlayerBar.tsx';

interface Props {
  dense?: boolean;
}

export default memo(function Layout({ dense }: Props) {
  const loaded = usePreloader((s) => s.pixiReady && s.assetsReady);
  useBoardRect();

  useEffect(() => {
    assetsReady.then(() => usePreloader.getState().setAssetsReady());
  }, []);

  return (
    <main id="playable-area" className={styles.playableArea} data-loaded={loaded || undefined} data-dense={dense || undefined}>
      <SvgSprite />
      <HiddenHeader />
      <div className={styles.playableContent}>
        <PlayerBar color="b" />
        <div className={styles.board}>
          <BoardControls />
          <GameBoard />
          <Annotation />
        </div>
        <PlayerBar color="w" />
        <VersusBanner />
        <CheckBanner />
      </div>
      <Vignette />
      <SoundEffects />
      <GameOver />
      <HeroAnimation />
    </main>
  );
});
