import { memo, useEffect } from 'react';
import HiddenHeader from './HiddenHeader';
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

export default memo(function Layout() {
  const loaded = usePreloader((s) => s.pixiReady && s.assetsReady);
  useBoardRect();

  useEffect(() => {
    assetsReady.then(() => usePreloader.getState().setAssetsReady());
  }, []);

  return (
    <main id="playable-area" className={styles.playableArea} data-loaded={loaded || undefined}>
      <HiddenHeader />
      <PlayerBar color="b" />
      <div className={styles.board}>
        <BoardControls />
        <GameBoard />
        <Annotation />
      </div>
      <PlayerBar color="w" />
      <VersusBanner />
      <CheckBanner />
      <Vignette />
      <SoundEffects />
      <GameOver />
      <HeroAnimation />
    </main>
  );
});
