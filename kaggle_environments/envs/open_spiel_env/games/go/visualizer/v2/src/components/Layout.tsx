import { memo } from 'react';
import HiddenHeader from './HiddenHeader.tsx';
import BoardControls from './BoardControls.tsx';
import GameBoard from './GameBoard';
import Annotation from './Annotation.tsx';
import ScorePanel from './ScorePanel';
import VersusBanner from './VersusBanner.tsx';
import GameOver from './GameOver';
import HeroAnimation from './HeroAnimation.tsx';
import SoundEffects from './SoundEffects.tsx';
import styles from './GameRenderer.module.css';

export default memo(function Layout() {
  console.log("Layout")
  return (
    <main id="go-playable-area" className={styles.playableArea}>
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
