import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { GameRenderer } from './renderer';
import { codenamesTransformer } from './codenamesTransformer';

/**
 * Main entry point for the Vite application.
 * Mounts the Kaggle Environment React core wrapper to the DOM.
 */
const app = document.getElementById('app');
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: 'codenames',
      GameRenderer: GameRenderer,
      transformer: codenamesTransformer,
      ui: 'side-panel',
    })
  );
}
