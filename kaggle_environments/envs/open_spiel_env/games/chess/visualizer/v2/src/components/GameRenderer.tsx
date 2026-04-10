import { memo, useEffect } from 'react';
import { Chess } from 'chess.js';
import { GameRendererProps } from '@kaggle-environments/core';
import { ChessStep, ChessPlayer } from '../transformers/chessReplayTypes';
import useGameStore from '../stores/useGameStore';
import Layout from './Layout';

export default memo(function GameRenderer(options: GameRendererProps<ChessStep[]>) {
  const setState = useGameStore((state) => state.setState);

  useEffect(() => {
    const step = options.replay.steps.at(options.step);

    if (!step) return;

    const history = options.replay.info!.stateHistory;
    const fen = history.at(Math.max(0, options.step - 1));
    const game = new Chess(fen);

    const player = step.players.find((player: ChessPlayer) => player.isTurn);
    if (player?.actionDisplayText) game.move(player.actionDisplayText);

    step.players.map((player, index) => {
      game.setHeader(index ? 'b' : 'w', player.name);
    });

    setState(game, options);
  }, [setState, options]);

  return <Layout />;
});
