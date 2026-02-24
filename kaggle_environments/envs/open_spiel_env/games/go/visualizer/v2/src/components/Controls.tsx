import { useEffect, useRef } from 'react';
import { Game } from 'tenuki';
import { createReplayVisualizer, ReplayAdapter, RendererOptions, GoStep } from '@kaggle-environments/core';
import useGoStore from '../stores/useGoStore';

export default function Controls() {
  const containerRef = useRef(null);
  const setState = useGoStore((state) => state.setState);

  useEffect(() => {
    const renderer = (options: RendererOptions<GoStep[]>) => {
      const step = options.replay.steps.at(options.step);

      console.log(options.step);

      const board = step!.boardState.board!;
      const size = step?.boardState.board_size!;

      const go = new Game({ boardSize: size });

      for (let x = 0; x < size; x++) {
        for (let y = 0; y < size; y++) {
          const color = board[x][y];

          if (color === '.') continue;

          if (go.currentPlayer() !== (color === 'B' ? 'black' : 'white')) go.pass();

          console.log(go.playAt(x, y));
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
