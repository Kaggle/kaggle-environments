import { useCallback } from 'react';
import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import GameRenderer from './components/GameRenderer';

export default function App() {
  const init = useCallback((element: HTMLDivElement | null) => {
    if (!element) return;
    const adapter = new ReplayAdapter({
      gameName: 'open_spiel_mancala',
      GameRenderer: GameRenderer as any,
      ui: 'side-panel',
    });
    createReplayVisualizer(element, adapter);
  }, []);

  return <div id="container" ref={init} style={{ width: '100%', height: '100%' }} />;
}
