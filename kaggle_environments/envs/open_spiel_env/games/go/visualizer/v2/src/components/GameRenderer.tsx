import { useEffect, memo } from 'react';
import { Game } from 'tenuki';
import { GameRendererProps } from '@kaggle-environments/core';
import { GoStep } from '../transformers/goReplayTypes';
import GameBoard from '../components/GameBoard';
import ScorePanel from '../components/ScorePanel';
import GameOverModal from '../components/GameOverModal';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences.ts';
import HeroAnimationModal from './HeroAnimationModal.tsx';
import VersusBanner from './VersusBanner.tsx';
import styles from './Gamerenderer.module.css';
import Notation from './Notation.tsx';
import { BoardControls } from './BoardControls.tsx';

export default memo(function GameRenderer(options: GameRendererProps<GoStep[]>) {
  const game = useGameStore((s) => s.game);
  const setState = useGameStore((state) => state.setState);
  const showHeroAnimations = usePreferences((s) => s.showHeroAnimations);
  const showAnnotations = usePreferences((state) => state.showAnnotations);

  useEffect(() => {
    const parameters = options.replay.configuration.openSpielGameParameters;
    const boardSize = parameters.board_size;
    const komi = parameters.komi;
    const scoring = 'area'; // Tromp-Tailor Rules
    const game = new Game({ boardSize, komi, scoring });

    for (let i = 0; i <= options.step; i++) {
      const step = options.replay.steps.at(i);
      const [, move] = step!.boardState.previous_move_a1!.split(' ');

      if (move === 'PASS') {
        game.pass();
      } else {
        type index = { [key: string]: number };
        const cols: index = {
          'a': 0,
          'b': 1,
          'c': 2,
          'd': 3,
          'e': 4,
          'f': 5,
          'g': 6,
          'h': 7,
          'j': 8,
          'k': 9,
          'l': 10,
          'm': 11,
          'n': 12,
          'o': 13,
          'p': 14,
          'q': 15,
          'r': 16,
          's': 17,
          't': 18,
        };
        const y = boardSize - parseInt(move.slice(1));
        const x = cols[move.charAt(0)];
        game.playAt(y, x);
      }
    }

    setState(game, options);
  }, [options, setState]);

  const gameOver = game.isOver();
  // React 18 doesn't support the `inert` HTML attribute as a prop, so we
  // set it imperatively via a ref callback. This can be replaced with a
  // regular `inert` prop once the project upgrades to React 19+.
  const inertRef = (el: HTMLElement | null) => {
    if (!el) return;
    if (gameOver) el.setAttribute('inert', '');
    else el.removeAttribute('inert');
  };

  return (
    <main id="go-playable-area" className={styles.playableArea}>
      <h1 className="visually-hidden">
        {options.replay.info?.TeamNames?.[0] ?? 'Black'} vs. {options.replay.info?.TeamNames?.[1] ?? 'White'}
      </h1>
      <div className={styles.board} ref={inertRef}>
        <BoardControls />
        <GameBoard />
        {showAnnotations && (
          <div className={styles.notationSlot} aria-live="polite">
            <Notation />
          </div>
        )}
      </div>
      <div ref={inertRef}>
        <ScorePanel />
      </div>
      {options.step === 0 && <VersusBanner options={options} />}
      {gameOver && <GameOverModal />}
      {showHeroAnimations && <HeroAnimationModal />}
    </main>
  );
});
