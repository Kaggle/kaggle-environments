import { useCallback } from 'react';
import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { transformer } from './transformers/transformer';
import { getStepRenderTime } from './utils/getStepRenderTime';
import { getStepLabel } from './utils/getStepLabel';
import { GAME_NAME } from './utils/analytics';
import GameRenderer from './components/GameRenderer';
import './App.css';

export default function App() {
  const init = useCallback((element: HTMLDivElement) => {
    const gameName = GAME_NAME;
    const ui = 'side-panel';
    const adapter = new ReplayAdapter({
      gameName,
      GameRenderer,
      ui,
      transformer,
      getStepRenderTime,
      getStepLabel,
    });
    createReplayVisualizer(element, adapter);
  }, []);

  return <div id="container" ref={init} />;
}
