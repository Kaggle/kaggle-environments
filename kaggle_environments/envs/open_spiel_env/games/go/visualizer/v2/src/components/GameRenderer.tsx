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
        const row: { [property: string]: number } = { a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, j: 9 };
        const x = row[move.charAt(0)] - 1;
        const y = parseInt(move.charAt(1)) - 1;
        game.playAt(x, y);
      }
    }

    console.log(game.currentState());
    // for (let x = 0; x < game.boardSize; x++) {
    //   for (let y = 0; y < game.boardSize; y++) {
    //     if (game.currentState().inAtari(x, y)) console.log('atari', x, y, game.currentState().intersectionAt(x, y).value);
    //   }
    // }

    console.log(game.currentState().intersections);
    console.log(game.intersections());

    const state = game.currentState();

    console.log(
      'atari',
      state.intersections.filter((intersection) => state.inAtari(intersection.x, intersection.y))
    );

    console.log(
      'atari',
      state.intersections.filter((intersection) => state.inAtari(intersection.x, intersection.y))
    );

    // console.log(game.territory());

    // console.log(game._scorer.territory(game));

    setState(game);
  }, [setState, options]);

  return <StyledBoard />;
}
