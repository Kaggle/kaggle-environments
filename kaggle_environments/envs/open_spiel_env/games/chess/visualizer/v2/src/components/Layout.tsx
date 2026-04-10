import { memo } from 'react';
import HiddenHeader from './HiddenHeader';
import BoardControls from './BoardControls';
import GameBoard from './GameBoard';
import Annotation from './Annotation';
import { SoundEffects } from './SoundEffects.tsx';
import VersusBanner from './VersusBanner';
import GameOver from './GameOver';
import HeroAnimation from './HeroAnimation';
import useBoardRect from '../hooks/useBoardRect';
import styles from './Layout.module.css';
import { Vignette } from './Vignette.tsx';
import PlayerBar from './PlayerBar.tsx';

export default memo(function Layout() {
  useBoardRect();

  return (
    <main id="playable-area" className={styles.playableArea} data-loaded={true}>
      <HiddenHeader />
      <PlayerBar color="b" />
      <div className={styles.board}>
        <BoardControls />
        <GameBoard />
        <Annotation />
      </div>
      <PlayerBar color="w" />
      <VersusBanner />
      <Vignette />
      <SoundEffects />
      <GameOver />
      <HeroAnimation />
    </main>
  );
});
