import { useCallback } from 'react';
import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { transformer } from './transformers/transformer';
import GameRenderer from './components/GameRenderer';
import './App.css';

export default function App() {
  const init = useCallback((element: HTMLDivElement) => {
    const gameName = 'open_spiel_chess';
    const ui = 'side-panel';
    const adapter = new ReplayAdapter({
      gameName,
      GameRenderer,
      ui,
      transformer,
    });
    createReplayVisualizer(element, adapter);
  }, []);

  return <div id="container" ref={init} />;
}
