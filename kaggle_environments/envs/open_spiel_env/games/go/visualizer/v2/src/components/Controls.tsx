import { useEffect, useRef } from 'react';
import { Game } from 'tenuki';
import { createReplayVisualizer, ReplayAdapter, RendererOptions, GoStep } from '@kaggle-environments/core';
import useGoStore from '../stores/useGoStore';

export default function Controls() {
  const containerRef = useRef(null);
  const setState = useGoStore((state) => state.setState);

  useEffect(() => {
    const renderer = (options: RendererOptions<GoStep[]>) => {
      const size = options.replay.configuration.openSpielGameParameters.board_size;
      const go = new Game({ boardSize: size });

      for (let i = 0; i <= options.step; i++) {
        const step = options.replay.steps.at(i);
        const colorAndMove = step!.boardState.previous_move_a1!;
        const move = colorAndMove.split(' ')[1]!;

        if (move === 'PASS') {
          go.pass();
        } else {
          const index: { [key: string]: number } = {
            '1': 0,
            '2': 1,
            '3': 2,
            '4': 3,
            '5': 4,
            '6': 5,
            '7': 6,
            '8': 7,
            '9': 8,
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7,
            'j': 8,
          };

          const x = index[move.charAt(0)];
          const y = index[move.charAt(1)];
          go.playAt(x, y);
        }
      }

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
