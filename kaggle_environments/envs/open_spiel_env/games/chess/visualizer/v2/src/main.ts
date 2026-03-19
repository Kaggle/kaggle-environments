import { createReplayVisualizer, ReplayAdapter, generateEaseInDelayDistribution } from '@kaggle-environments/core';
import { renderer } from './chess_renderer';
import { chessTransformer, getChessStepLabel, getChessStepDescription } from './transformers/chessTransformer';
import { ChessStep } from './transformers/chessReplayTypes';

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
    gameName: 'open_spiel_chess',
    renderer,
    ui: 'inline',
    transformer: (replay) => ({
      ...replay,
      steps: chessTransformer(replay),
      isTransformed: true,
    }),
    getStepLabel: (step) => getChessStepLabel(step as ChessStep),
    getStepDescription: (step) => getChessStepDescription(step as ChessStep),
    getTokenRenderDistribution: generateEaseInDelayDistribution,
  })
);
