import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
import { getOrbitWarsStepRenderTime } from './timing';
import './style.css';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

if (import.meta.env?.DEV && import.meta.hot) {
  import.meta.hot.accept();
}

createReplayVisualizer(
  app,
  new ReplayAdapter({
    gameName: 'orbit_wars',
    renderer: renderer as any,
    ui: 'side-panel',
    getStepRenderTime: (step, replayMode, speedModifier) => getOrbitWarsStepRenderTime(step, replayMode, speedModifier),
  })
);
