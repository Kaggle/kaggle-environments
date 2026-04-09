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
    const player = step!.players.find((player: ChessPlayer) => player.isTurn);

    const history = options.replay.info!.stateHistory;
    const fen = history[options.step - 1] ?? undefined;
    const game = new Chess(fen);

    if (player?.actionDisplayText) game.move(player.actionDisplayText);

    step!.players.map((p, i) => {
      const color = i ? 'b' : 'w';
      game.setHeader(color, p.name);
    });

    setState(game, options);
  }, [setState, options]);

  return <Layout />;
});
