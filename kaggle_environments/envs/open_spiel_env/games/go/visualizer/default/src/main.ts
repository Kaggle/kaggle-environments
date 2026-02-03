import { createReplayVisualizer, LegacyAdapter, processEpisodeData } from '@kaggle-environments/core';
import { renderer } from './renderer';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

const adapter = new LegacyAdapter(renderer as any);
if (app) {
  // Set up an HMR boundary for development
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(app, adapter, {
    transformer: (replay) => processEpisodeData(replay, 'open_spiel_go'),
  });
}
