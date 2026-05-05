import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
import { loaTransformer } from './transformers/loaTransformer';
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
    gameName: 'open_spiel_lines_of_action',
    renderer: renderer as any,
    ui: 'side-panel',
    transformer: (replay) => ({
      ...replay,
      steps: loaTransformer(replay),
      isTransformed: true,
    }),
  })
);
