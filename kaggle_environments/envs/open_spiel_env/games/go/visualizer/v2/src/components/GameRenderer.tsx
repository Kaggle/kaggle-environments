import { useEffect } from 'react';
import { createGame } from 'jgoboard';
import { GoStep, GameRendererProps } from '@kaggle-environments/core';
import StyledBoard from '../components/StyledBoard';
import useGoStore from '../stores/useGoStore';

export default function GameRenderer(options: GameRendererProps<GoStep[]>) {
  const setState = useGoStore((state) => state.setState);

  useEffect(() => {
    const go = createGame({ size: 9 });

    for (let i = 0; i <= options.step; i++) {
      const step = options.replay.steps.at(i);
      const [, move] = step!.boardState.previous_move_a1!.split(' ');

      if (move === 'PASS') {
        go.pass();
      } else {
        go.play(move);
      }
    }

    setState(go);
  }, [setState, options]);

  return <StyledBoard />;
}
