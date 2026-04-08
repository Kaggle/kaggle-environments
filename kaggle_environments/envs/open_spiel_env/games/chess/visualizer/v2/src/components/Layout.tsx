import { memo } from 'react';
import HiddenHeader from './HiddenHeader';
import BoardControls from './BoardControls';
import GameBoard from './GameBoard';
import Annotation from './Annotation';
import styles from './Layout.module.css';

export default memo(function Layout() {
  return (
    <main id="playable-area" className={styles.playableArea} data-loaded={true}>
      <HiddenHeader />
      <div className={styles.board}>
        <BoardControls />
        <GameBoard />
        <Annotation />
      </div>
    </main>
  );
});
