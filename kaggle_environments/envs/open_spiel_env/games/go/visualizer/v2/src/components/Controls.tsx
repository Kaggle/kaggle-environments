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

        if (move === 'PASS') {
          go.pass();
        } else {
          go.play(move);
        }
      }

      // Check against game state captured by OpenSpiel. Matches.
      // console.log(options.replay.steps.at(options.step)?.boardState.board);

      // Log the llm thoughts and player move. Note the agent names aren't
      // returned in the same way as for chess, and they don't quite seem
      // real in the replay data either.
      const step = options.replay.steps.at(options.step);
      const player = step!.players.find((player) => player.isTurn);
      const [color, move] = step!.boardState.previous_move_a1!.split(' ');
      console.log(player?.thoughts);
      console.log(`${player?.name} (${color}): ${move}`);

      setState(go);
    };

    const container = containerRef.current!;
    const gameName = 'open_spiel_go';
    const ui = 'side-panel';
    const adapter = new ReplayAdapter({ gameName, renderer, ui });

    createReplayVisualizer(container, adapter);
  }, [setState]);

  return <div id="controls" ref={containerRef} />;
}
