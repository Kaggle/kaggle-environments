import { memo } from 'react';
import HiddenHeader from './HiddenHeader';
import BoardControls from './BoardControls';
import GameBoard from './GameBoard';
import Annotation from './Annotation';
import ScorePanel from './ScorePanel';
import VersusBanner from './VersusBanner';
import GameOver from './GameOver';
import HeroAnimation from './HeroAnimation';
import SoundEffects from './SoundEffects';
import styles from './Layout.module.css';

export default memo(function Layout() {
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
