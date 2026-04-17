import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';

const app = document.getElementById('root');
if (!app) {
  throw new Error('Could not find root element');
}

createReplayVisualizer(
  app,
  new ReplayAdapter({
    gameName: 'open_spiel_go',
    renderer: () => {},
    ui: 'inline',
  })
);
