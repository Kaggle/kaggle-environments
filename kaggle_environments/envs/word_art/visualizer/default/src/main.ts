import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
import { wordArtTransformer } from './wordArtTransformer';
import './style.css';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find #app element');
}

if (import.meta.env?.DEV && import.meta.hot) {
  import.meta.hot.accept();
}

createReplayVisualizer(
  app,
  new ReplayAdapter({
    gameName: 'word_art',
    renderer: renderer as any,
    // Populates the side-panel step log with per-player action text
    // and LLM thoughts. Without a transformer the sidebar shows a bare
    // step counter with no per-turn detail.
    transformer: wordArtTransformer,
    ui: 'side-panel',
  })
);
