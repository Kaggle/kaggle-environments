import { LegacyRendererOptions } from '@kaggle-environments/core';

const MAX_WIDTH = 960;
const MAX_HEIGHT = 280;

const PADDING = 20;

const LABEL_Y = 40;
const SIGN_ID_Y = 80;
const RESULT_Y = 120;
const SCORE_Y = 160;

export function renderer(options: LegacyRendererOptions) {
  const { steps, step, parent, replay, width = 400, height = 400 } = options;

  const environment = {
    steps: steps,
    info: { TeamNames: replay.info?.TeamNames || ['Player 1', 'Player 2'] },
  };

  // Canvas Setup.
  let canvas = parent.querySelector('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    parent.appendChild(canvas);
  }

  // Set display size (css pixels)
  canvas.style.width = `${Math.min(MAX_WIDTH, width)}px`;
  canvas.style.height = `${Math.min(MAX_HEIGHT, height)}px`;

  // Get the device pixel ratio
  const dpr = window.devicePixelRatio || 1;

  // Set actual size in memory (scaled for device pixel ratio)
  canvas.width = Math.min(MAX_WIDTH, width) * dpr;
  canvas.height = Math.min(MAX_HEIGHT, height) * dpr;

  // Canvas setup and reset.
  const c = canvas.getContext('2d');
  canvas.width = Math.min(MAX_WIDTH, width);
  canvas.height = Math.min(MAX_HEIGHT, height);

  if (!c) {
    return;
  }

  c.clearRect(0, 0, canvas.width, canvas.height);

  if (step < environment.steps.length - 1) {
    const state: any = environment.steps[step + 1];
    const last_state: any = environment.steps[step];

    const p1_move = state[0].observation.lastActions[0];
    const p2_move = state[0].observation.lastActions[1];

    const info = environment.info;
    const player1_text = info?.TeamNames?.[0] || 'Player 1';
    const player2_text = info?.TeamNames?.[1] || 'Player 2';

    const ctx = canvas.getContext('2d');

    if (!ctx) {
      return;
    }

    const row_width = (Math.min(MAX_WIDTH, width) - PADDING * 2) / 3;
    const label_x = PADDING;
    const player1_x = PADDING + row_width;
    const player2_x = PADDING + 2 * row_width;

    ctx.font = '30px sans-serif';
    ctx.fillStyle = '#FFFFFF';

    // Player Row
    ctx.fillText(player1_text, player1_x, LABEL_Y);
    ctx.fillText(player2_text, player2_x, LABEL_Y);

    // Action Id Row
    ctx.fillText('Action:', label_x, SIGN_ID_Y);
    ctx.fillText(p1_move, player1_x, SIGN_ID_Y);
    ctx.fillText(p2_move, player2_x, SIGN_ID_Y);

    // Result Row
    ctx.fillText('Result:', label_x, RESULT_Y);
    if (state[0].reward - last_state[0].reward > 0) {
      ctx.fillText('Win', player1_x, RESULT_Y);
    }

    if (state[1].reward - last_state[1].reward > 0) {
      ctx.fillText('Win', player2_x, RESULT_Y);
    }

    // Reward Row
    ctx.fillText('Reward:', label_x, SCORE_Y);
    ctx.fillText(state[0].reward, player1_x, SCORE_Y);
    ctx.fillText(state[1].reward, player2_x, SCORE_Y);
  }
}
