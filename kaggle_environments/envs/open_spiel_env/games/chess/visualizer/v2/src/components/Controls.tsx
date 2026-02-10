import { useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import {
  createReplayVisualizer,
  processEpisodeData,
  LegacyAdapter,
  LegacyRendererOptions,
  ReplayData,
  ChessPlayer,
  ChessStep,
} from '@kaggle-environments/core';
import useChessStore from '../stores/useChessStore';

const Controls = () => {
  const containerRef = useRef(null);
  const setState = useChessStore((state) => state.setState);

  useEffect(() => {
    const renderer = (options: LegacyRendererOptions<ChessStep[]>) => {
      const step = options.steps.at(options.step);
      const player = step!.players.find((player: ChessPlayer) => player.isTurn);

      if (player) {
        const history = options.replay.info!.stateHistory;
        const fen = history.at(options.step);
        const chess = new Chess(fen);
        const move = chess.move(player.actionDisplayText!);
        chess.setHeader('name', player.name);

        console.log(player.thoughts);
        console.log(`${player.name} (${move.color}): ${move.piece} ${move.from} -> ${move.to}`);

        setState(chess);
      }
    };

    const container = containerRef.current!;
    const adapter = new LegacyAdapter<ChessStep[]>(renderer);
    const transformer = (replay: ReplayData) => processEpisodeData(replay, 'open_spiel_chess');

    createReplayVisualizer(container, adapter, { transformer });
  }, [setState]);

  return <div id="controls" ref={containerRef} />;
};

export default Controls;
