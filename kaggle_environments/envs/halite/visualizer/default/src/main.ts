import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './halite_renderer.js';

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
    gameName: 'halite',
    renderer: renderer as any,
    ui: 'inline',
  })
);
