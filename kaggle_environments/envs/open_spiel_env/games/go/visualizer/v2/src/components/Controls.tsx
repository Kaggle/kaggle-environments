import { useEffect, useRef } from 'react';
import { createReplayVisualizer, ReplayAdapter, RendererOptions, GoStep } from '@kaggle-environments/core';

export default function Controls() {
  const containerRef = useRef(null);

  useEffect(() => {
    const renderer = (options: RendererOptions<GoStep[]>) => {
      const step = options.replay.steps.at(options.step);
      const player = step!.players.find((player) => player.isTurn);
      console.log(player);
    };

    const container = containerRef.current!;
    const gameName = 'open_spiel_go';
    const ui = 'inline';
    const adapter = new ReplayAdapter({ gameName, renderer, ui });

    createReplayVisualizer(container, adapter);
  }, []);

  return <div id="controls" ref={containerRef} />;
}
