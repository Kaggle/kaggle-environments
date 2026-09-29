import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
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
    gameName: 'pyxis',
    renderer: renderer as any,
    // Pyxis agents submit numeric action heads, not prose, so there is nothing
    // for the reasoning sidebar to show.
    ui: 'inline',
  })
);
