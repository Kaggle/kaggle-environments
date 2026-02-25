import { useEffect, useRef } from 'react';
import GameRenderer from './components/GameRenderer';
import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import './App.css';

function App() {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current!;
    const gameName = 'open_spiel_go';
    const ui = 'side-panel';
    const adapter = new ReplayAdapter({ gameName, GameRenderer, ui });

    createReplayVisualizer(container, adapter);
  });

  return <div id="container" ref={containerRef} />;
}

export default App;
