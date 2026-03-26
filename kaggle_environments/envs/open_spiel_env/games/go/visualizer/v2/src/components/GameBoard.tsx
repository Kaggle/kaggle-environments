import { CellValue } from '../types/game.ts';
import { GoBoard } from './GoBoard';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';

export default function GameBoard() {
  const game = useGameStore((state) => state.game);
  const showTerritory = usePreferences((state) => state.showTerritory);
  const reducedMotion = usePreferences((state) => state.reducedMotion);

  const state = game.currentState();
  const size = game.boardSize;
  const step = state.moveNumber;

  const cells: Record<string, CellValue> = { black: 'B', white: 'W', empty: '.' };
  const grid: CellValue[][] = Array.from({ length: size }, () => new Array(size));
  for (const item of state.intersections) {
    grid[item.y][item.x] = cells[item.value ?? 'empty'];
  }

  const played = state.playedPoint ? { row: state.playedPoint.y, col: state.playedPoint.x } : null;

  const scorer = game._scorer.territory(game);
  const territory = {
    black: step === 1 ? [] : scorer.black.map((point: { x: number; y: number }) => ({ row: point.y, col: point.x })),
    white: step === 1 ? [] : scorer.white.map((point: { x: number; y: number }) => ({ row: point.y, col: point.x })),
  };

  const atari = state.intersections
    .filter((intersection) => state.inAtari(intersection.y, intersection.x))
    .map((intersection) => ({ row: intersection.y, col: intersection.x }));

  // React 18 doesn't support the `inert` HTML attribute as a prop, so we
  // set it imperatively via a ref callback. This can be replaced with a
  // regular `inert` prop once the project upgrades to React 19+.
  const inertRef = (el: HTMLElement | null) => {
    if (!el) return;
    if (game.gameOver) el.setAttribute('inert', '');
    else el.removeAttribute('inert');
  };

  return (
    <div id="board" ref={inertRef}>
      <GoBoard
        boardSize={size}
        grid={grid}
        step={step}
        lastPlayed={played}
        atari={atari}
        territory={showTerritory ? territory : { black: [], white: [] }}
        reducedMotion={reducedMotion}
      />
    </div>
  );
}
