import { useEffect } from 'react';
import { Game } from 'tenuki';
import { GameRendererProps } from '@kaggle-environments/core';
import { GoStep } from '../transformers/goReplayTypes';
import GameBoard from '../components/GameBoard';
import ScorePanel from '../components/ScorePanel';
import CapturePots from '../components/CapturePots';
import GameOverModal from '../components/GameOverModal';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences.ts';
import knightRiv from '../assets/kaggle_knight.riv?url';
import queenRiv from '../assets/kaggle_queen.riv?url';
import HeroAnimationModal from './HeroAnimationModal.tsx';
import VersusBanner from './VersusBanner.tsx';

export default function GameRenderer(options: GameRendererProps<GoStep[]>) {
  const game = useGameStore((s) => s.game);
  const setState = useGameStore((state) => state.setState);
  const showHeroAnimations = usePreferences((s) => s.showHeroAnimations);

  useEffect(() => {
    // Seems like these OpenSpiel 13x13 log files where the steps is empty are
    // broken and shouldn't be listed on Game Arena
    if (options.replay.steps.length === 0) return;

    const parameters = options.replay.configuration.openSpielGameParameters;
    // OpenSpiel parameter incorrectly set for board size in example replays
    // const boardSize = parameters.board_size;
    const boardSize = options.replay.steps[0].boardState.board_size;
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

  return (
    <div id="go-playable-area">
      <link rel="preload" href={knightRiv} as="fetch" crossOrigin="anonymous" />
      <link rel="preload" href={queenRiv} as="fetch" crossOrigin="anonymous" />
      <GameBoard />
      <div>
        <ScorePanel />
        <CapturePots />
      </div>
      {options.step === 0 && <VersusBanner options={options} />}
      {game.isOver() && <GameOverModal />}
      {showHeroAnimations && <HeroAnimationModal />}
    </div>
  );
}
