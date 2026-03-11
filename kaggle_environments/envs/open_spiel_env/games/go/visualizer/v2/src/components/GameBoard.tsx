import { memo } from 'react';
import { CellValue } from '../types/game.ts';
import { FeatureToggles } from './FeatureToggles.tsx';
import { GoBoard } from './GoBoard';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import { tenukiLogger } from '../utils/tenukiLogger';
import styles from './GameBoard.module.css';
import { WithPopover } from './WithPopover.tsx';

export default memo(function GameBoard() {
  const game = useGameStore((state) => state.game);
  const showTerritory = usePreferences((state) => state.showTerritory);
  const reducedMotion = usePreferences((state) => state.reducedMotion);

  tenukiLogger(game);

  const state = game.currentState();
  const size = game.boardSize;
  const step = state.moveNumber;

  const cells: Record<string, CellValue> = { black: 'B', white: 'W', empty: '.' };
  const grid: CellValue[][] = Array.from({ length: size }, () => new Array(size));
  for (const item of state.intersections) {
    grid[item.y][item.x] = cells[item.value ?? 'empty'];
  }

  let played = null;
  if (state.playedPoint) {
    played = {
      row: state.playedPoint.y,
      col: state.playedPoint.x,
    };
  }

  const captures = {
    black: state.blackStonesCaptured,
    white: state.whiteStonesCaptured,
  };

  const scorer = game._scorer.territory(game);
  const territory = {
    black: step === 1 ? [] : scorer.black.map((point: { x: number; y: number }) => ({ row: point.y, col: point.x })),
    white: step === 1 ? [] : scorer.white.map((point: { x: number; y: number }) => ({ row: point.y, col: point.x })),
  };

  const atari = state.intersections
    .filter((intersection) => state.inAtari(intersection.y, intersection.x))
    .map((intersection) => ({ row: intersection.y, col: intersection.x }));

  return (
    <div id="board">
      <div className={styles.boardControls}>
        <WithPopover id="info" icon="info">
          <p>
            Go is the ancient two-player game in which players attempt to control more territory on a grid by
            strategically placing black or white stones. Fun fact: unlike chess, black plays first.
          </p>
        </WithPopover>
        <WithPopover id="settings" icon="settings">
          <FeatureToggles />
        </WithPopover>
      </div>
      <div className={styles.boardAnchor}>
        <GoBoard
          boardSize={size}
          grid={grid}
          step={step}
          lastPlayed={played}
          captures={captures}
          atari={atari}
          territory={showTerritory ? territory : { black: [], white: [] }}
          reducedMotion={reducedMotion}
        />
      </div>
      <div></div>
    </div>
  );
});
