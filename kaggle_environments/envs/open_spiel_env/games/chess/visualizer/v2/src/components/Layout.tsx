import { memo } from 'react';
import HiddenHeader from './HiddenHeader';
import BoardControls from './BoardControls';
import GameBoard from './GameBoard';
import Annotation from './Annotation';
import VersusBanner from './VersusBanner';
import GameOver from './GameOver';
import HeroAnimation from './HeroAnimation';
import useBoardRect from '../hooks/useBoardRect';
import styles from './Layout.module.css';
import { Vignette } from './Vignette.tsx';

export default memo(function Layout() {
  useBoardRect();

  return (
    <main id="playable-area" className={styles.playableArea} data-loaded={true}>
      <HiddenHeader />
      <div className={styles.board}>
        <BoardControls />
        <GameBoard />
        <Annotation />
      </div>
      <VersusBanner />
      <Vignette />
      <GameOver />
      <HeroAnimation />
    </main>
  );
});
