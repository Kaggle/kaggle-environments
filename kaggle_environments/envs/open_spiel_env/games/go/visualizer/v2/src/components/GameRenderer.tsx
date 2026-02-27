import { useEffect } from 'react';
import { createGame } from 'jgoboard';
import { GoStep, GameRendererProps } from '@kaggle-environments/core';
import StyledBoard from '../components/StyledBoard';
import useGameStore from '../stores/useGameStore';

export default function GameRenderer(options: GameRendererProps<GoStep[]>) {
  const setState = useGameStore((state) => state.setState);

  useEffect(() => {
    const game = createGame({ size: 9 });

    for (let i = 0; i <= options.step; i++) {
      const step = options.replay.steps.at(i);
      const [, move] = step!.boardState.previous_move_a1!.split(' ');

      if (move === 'PASS') {
        game.pass();
      } else {
        game.play(move);
      }
    }

    setState(game);
  }, [setState, options]);

  return <StyledBoard />;
}
