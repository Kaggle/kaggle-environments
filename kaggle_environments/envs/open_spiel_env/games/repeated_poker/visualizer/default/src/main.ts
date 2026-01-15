import { createReplayVisualizer, LegacyAdapter, processEpisodeData } from '@kaggle-environments/core';
import { renderer } from './repeated_poker_renderer';
import './style.css';

const app = document.getElementById('app');
if (app) {
  // Set up an HMR boundary for development
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }

  createReplayVisualizer(app, new LegacyAdapter(renderer), {
    transformer: (replay) => processEpisodeData(replay, 'open_spiel_repeated_poker'),
  });
}
