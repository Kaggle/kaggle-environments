import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './ipd_renderer';
import './style.css';

const app = document.getElementById('app');
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: 'ipd',
      renderer: renderer,
      ui: 'side-panel',
    })
  );
}
