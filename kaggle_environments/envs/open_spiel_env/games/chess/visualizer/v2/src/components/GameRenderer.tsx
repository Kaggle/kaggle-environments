import { useEffect } from 'react';
import { Chess } from 'chess.js';
import { ChessPlayer, ChessStep, GameRendererProps } from '@kaggle-environments/core';
import StyledBoard from '../components/StyledBoard';
import Legend from '../components/Legend';
import Meter from '../components/Meter';
import Openings from '../components/Openings';
import useGameStore from '../stores/useGameStore';

export default function GameRenderer(options: GameRendererProps<ChessStep[]>) {
  const setState = useGameStore((state) => state.setState);

  useEffect(() => {
    const step = options.replay.steps.at(options.step);
    const player = step!.players.find((player: ChessPlayer) => player.isTurn);

    if (player) {
      const history = options.replay.info!.stateHistory;
      const fen = history.at(options.step);
      const game = new Chess(fen);
      const move = game.move(player.actionDisplayText!);

      step!.players.forEach((p) => {
        const opposite = { w: 'b', b: 'w' };
        const color = p.isTurn ? move.color : opposite[move.color];
        game.setHeader(color, p.name);
      });

      setState(game);
    }
  }, [setState, options]);

  return (
    <div id="renderer">
      <Meter />
      <StyledBoard />
      <Legend />
      <Openings />
    </div>
  );
}
