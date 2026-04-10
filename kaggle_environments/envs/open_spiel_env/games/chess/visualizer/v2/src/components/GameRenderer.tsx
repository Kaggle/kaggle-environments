import { memo, useEffect } from 'react';
import { Chess } from 'chess.js';
import { GameRendererProps } from '@kaggle-environments/core';
import { ChessStep } from '../transformers/chessReplayTypes';
import useGameStore from '../stores/useGameStore';
import Layout from './Layout';

export default memo(function GameRenderer(options: GameRendererProps<ChessStep[]>) {
  const setState = useGameStore((state) => state.setState);

  useEffect(() => {
    const game = new Chess();

    for (const step of options.replay.steps) {
      if (step.step > options.step) break;
      const move = step.players.find((p) => p.isTurn)?.actionDisplayText;
      if (move) game.move(move);
    }

    game.setHeader('b', options.replay.info?.TeamNames.at(0));
    game.setHeader('w', options.replay.info?.TeamNames.at(1));

    setState(game, options);
  }, [setState, options]);

  return <Layout />;
});
