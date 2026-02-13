import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './chess_renderer';

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
    gameName: 'open_spiel_chess',
    renderer,
    ui: 'inline',
  })
);
