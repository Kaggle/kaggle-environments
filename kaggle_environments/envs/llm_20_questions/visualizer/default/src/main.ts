import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './llm_20_questions_renderer';

const app = document.getElementById('app');
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: 'llm_20_questions',
      renderer: renderer,
      ui: 'side-panel',
    })
  );
}
