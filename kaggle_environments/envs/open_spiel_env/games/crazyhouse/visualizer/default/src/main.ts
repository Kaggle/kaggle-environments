import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './crazyhouse_renderer';

const app = document.getElementById('app');
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: 'open_spiel_crazyhouse',
      renderer: renderer as any,
      ui: 'side-panel',
    })
  );
}
