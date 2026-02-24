import { useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import {
  createReplayVisualizer,
  ReplayAdapter,
  RendererOptions,
  ChessPlayer,
  ChessStep,
} from '@kaggle-environments/core';
import useChessStore from '../stores/useChessStore';

export default function Controls() {
  const containerRef = useRef(null);
  const setState = useChessStore((state) => state.setState);

  useEffect(() => {
    const renderer = (options: RendererOptions<ChessStep[]>) => {
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

        console.log(player.thoughts);
        console.log(`${player.name} (${move.color}): ${move.piece} ${move.from} -> ${move.to}`);

        setState(chess);
      }
    };

    const container = containerRef.current!;
    const gameName = 'open_spiel_chess';
    const ui = 'inline';
    const adapter = new ReplayAdapter<ChessStep[]>({ gameName, renderer, ui });

    createReplayVisualizer(container, adapter);
  }, [setState]);

  return <div id="controls" ref={containerRef} />;
}
