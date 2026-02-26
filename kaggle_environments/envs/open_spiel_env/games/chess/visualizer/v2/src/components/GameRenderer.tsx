import { useEffect } from 'react';
import { Chess } from 'chess.js';
import { ChessPlayer, ChessStep, GameRendererProps } from '@kaggle-environments/core';
import StyledBoard from '../components/StyledBoard';
import Legend from './Legend';
import useGoStore from '../stores/useChessStore';

export default function GameRenderer(options: GameRendererProps<ChessStep[]>) {
  const setState = useGoStore((state) => state.setState);

  useEffect(() => {
    const step = options.replay.steps.at(options.step);
    const player = step!.players.find((player: ChessPlayer) => player.isTurn);

    if (player) {
      const history = options.replay.info!.stateHistory;
      const fen = history.at(options.step);
      const chess = new Chess(fen);
      const move = chess.move(player.actionDisplayText!);

      step!.players.map((p) => {
        const opposite = { w: 'b', b: 'w' };
        const color = p.isTurn ? move.color : opposite[move.color];
        chess.setHeader(color, p.name);
      });

      setState(chess);
    }
  }, [setState, options]);

  return (
    <div id="renderer">
      <StyledBoard />
      <Legend />
    </div>
  );
}
