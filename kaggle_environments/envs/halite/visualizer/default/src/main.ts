import { createReplayVisualizer, LegacyAdapter } from '@kaggle-environments/core';
import { renderer } from './halite_renderer.js';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

if (app) {
  // Set up an HMR boundary for development
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(app, new LegacyAdapter(renderer as any));
}
