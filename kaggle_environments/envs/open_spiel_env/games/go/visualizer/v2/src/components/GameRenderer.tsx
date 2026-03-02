import { useEffect } from 'react';
import { Game } from 'tenuki';
import { GoStep, GameRendererProps } from '@kaggle-environments/core';
import StyledBoard from '../components/StyledBoard';
import useGameStore from '../stores/useGameStore';

export default function GameRenderer(options: GameRendererProps<GoStep[]>) {
  const setState = useGameStore((state) => state.setState);

  useEffect(() => {
    const game = new Game({ boardSize: 9 });

    for (let i = 0; i <= options.step; i++) {
      const step = options.replay.steps.at(i);
      const [, move] = step!.boardState.previous_move_a1!.split(' ');

      if (move === 'PASS') {
        game.pass();
      } else {
        // const row: [string: number] = { a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, j: 9 }
        game.playAt(0, 0);
      }
    }

    setState(game);
  }, [setState, options]);

  return <StyledBoard />;
}
