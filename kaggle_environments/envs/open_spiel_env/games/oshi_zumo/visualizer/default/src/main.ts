import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
import { oshiZumoTransformer } from './transformers/oshiZumoTransformer';
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
    gameName: 'open_spiel_oshi_zumo',
    renderer: renderer as any,
    ui: 'side-panel',
    transformer: (replay) => ({
      ...replay,
      steps: oshiZumoTransformer(replay),
      isTransformed: true,
    }),
  })
);
