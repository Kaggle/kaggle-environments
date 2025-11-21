import { createReplayVisualizer, LegacyAdapter, processEpisodeData } from '@kaggle-environments/core';
import { renderer } from './llm_20_questions_renderer';

const app = document.getElementById('app');
if (app) {
  // Set up an HMR boundary for development
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(app, new LegacyAdapter(renderer), {
    transformer: (replay) => processEpisodeData(replay, 'llm_20_questions'),
  });
}