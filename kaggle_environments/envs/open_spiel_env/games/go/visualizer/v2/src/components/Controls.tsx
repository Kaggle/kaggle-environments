import { useEffect, useRef } from 'react';
import { createGame } from 'jgoboard';
import { createReplayVisualizer, ReplayAdapter, RendererOptions, GoStep } from '@kaggle-environments/core';
import useGoStore from '../stores/useGoStore';

export default function Controls() {
  const containerRef = useRef(null);
  const setState = useGoStore((state) => state.setState);

  useEffect(() => {
    const renderer = (options: RendererOptions<GoStep[]>) => {
      const go = createGame({ size: 9 });

      for (let i = 0; i <= options.step; i++) {
        const step = options.replay.steps.at(i);
        const colorAndMove = step!.boardState.previous_move_a1!;
        const [, move] = colorAndMove.split(' ');

        move === 'PASS' ? go.pass() : go.play(move);
      }

      // Check against game state captured by OpenSpiel. Matches.
      // console.log(options.replay.steps.at(options.step)?.boardState.board);

      const step = options.replay.steps.at(options.step);
      const player = step!.players.find((player) => player.isTurn);
      console.log(player?.thoughts);

      setState(go);
    };

    const container = containerRef.current!;
    const gameName = 'open_spiel_go';
    const ui = 'inline';
    const adapter = new ReplayAdapter({ gameName, renderer, ui });

    createReplayVisualizer(container, adapter);
  }, []);

  return <div id="controls" ref={containerRef} />;
}
