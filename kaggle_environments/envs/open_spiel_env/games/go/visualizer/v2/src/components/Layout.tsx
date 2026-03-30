import { memo, useEffect } from 'react';
import HiddenHeader from './HiddenHeader';
import BoardControls from './BoardControls';
import GameBoard from './GameBoard';
import Annotation from './Annotation';
import ScorePanel from './ScorePanel';
import VersusBanner from './VersusBanner';
import GameOver from './GameOver';
import HeroAnimation from './HeroAnimation';
import SoundEffects from './SoundEffects';
import usePreloader from '../stores/usePreloader';
import useBoardRect from '../hooks/useBoardRect';
import { assetsReady } from '../utils/preloadAssets';
import styles from './Layout.module.css';

export default memo(function Layout() {
  const loaded = usePreloader((s) => s.pixiReady && s.assetsReady);
  useBoardRect();

  useEffect(() => {
    assetsReady.then(() => usePreloader.getState().setAssetsReady());
  }, []);

  return (
    <main id="go-playable-area" className={styles.playableArea} data-loaded={loaded || undefined}>
      <HiddenHeader />
      <div className={styles.board}>
        <BoardControls />
        <GameBoard />
        <Annotation />
      </div>
      <ScorePanel />
      <VersusBanner />
      <GameOver />
      <HeroAnimation />
      <SoundEffects />
    </main>
  );
});
