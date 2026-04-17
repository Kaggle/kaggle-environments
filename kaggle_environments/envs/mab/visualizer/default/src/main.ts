import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './mab_renderer';

const app = document.getElementById('app');
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: 'mab',
      renderer: renderer,
      ui: 'inline',
    })
  );
}
