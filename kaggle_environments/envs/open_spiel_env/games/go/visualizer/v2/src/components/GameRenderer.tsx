import { useEffect } from 'react';
import { Game } from 'tenuki';
import { GoStep, GameRendererProps } from '@kaggle-environments/core';
import StyledBoard from '../components/StyledBoard';
import useGameStore from '../stores/useGameStore';

export default function GameRenderer(options: GameRendererProps<GoStep[]>) {
  const setState = useGameStore((state) => state.setState);

  useEffect(() => {
    const parameters = options.replay.configuration.openSpielGameParameters;
    const boardSize = parameters.board_size;
    const komi = parameters.komi;
    const game = new Game({ boardSize, komi });

    for (let i = 0; i <= options.step; i++) {
      const step = options.replay.steps.at(i);
      const [, move] = step!.boardState.previous_move_a1!.split(' ');

      if (move === 'PASS') {
        game.pass();
      } else {
        type index = { [key: string]: number };
        const rows: index = { '1': 8, '2': 7, '3': 6, '4': 5, '5': 4, '6': 3, '7': 2, '8': 1, '9': 0 };
        const cols: index = { 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'j': 8 };
        const y = rows[move.charAt(1)];
        const x = cols[move.charAt(0)];
        game.playAt(y, x);
      }
    }

    setState(game, options);
  }, [setState, options]);

  return <StyledBoard />;
}
