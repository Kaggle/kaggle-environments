import { createReplayVisualizer, PreactAdapter, processEpisodeData } from '@kaggle-environments/core';
import { Renderer } from './renderer';
import './style.css';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

// TODO - fix this any when we figure out a global format
const adapter = new PreactAdapter(Renderer as any);
if (app) {
  // Set up an HMR boundary for development
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(app, adapter, {
    transformer: (replay) => processEpisodeData(replay, 'open_spiel_connect_four'),
  });
}
