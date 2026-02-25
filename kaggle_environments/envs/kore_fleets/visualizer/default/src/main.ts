import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './kore_fleets_renderer';

const app = document.getElementById('app');
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: 'kore_fleets',
      renderer: renderer as any,
      ui: 'inline',
    })
  );
}
