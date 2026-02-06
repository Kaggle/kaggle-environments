import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './repeated_poker_renderer';
import './style.css';

const app = document.getElementById('app');
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: 'open_spiel_repeated_poker',
      renderer: renderer,
      ui: 'side-panel',
    })
  );
}
