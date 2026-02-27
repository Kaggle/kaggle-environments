import { RendererOptions } from '@kaggle-environments/core';
import { SIGN_ICONS, SIGN_NAMES } from './consts';

const maxWidth = 960;
const maxHeight = 280;
const label_y = 40;
const sign_id_y = 80;
const sign_name_y = 120;
const sign_icon_y = 160;
const result_y = 200;
const score_y = 240;

export function renderer(context: RendererOptions) {
  const { replay, parent, step } = context;
  const steps = replay.steps;
  const width = parent.clientWidth || 400;
  const height = parent.clientHeight || 400;

  // Canvas Setup.
  let canvas = parent.querySelector('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    parent.appendChild(canvas);
  }

  // Set display size (css pixels)
  canvas.style.width = `${Math.min(maxWidth, width)}px`;
  canvas.style.height = `${Math.min(maxHeight, height)}px`;

  // Get the device pixel ratio
  const dpr = window.devicePixelRatio || 1;

  // Set actual size in memory (scaled for device pixel ratio)
  canvas.width = Math.min(maxWidth, width) * dpr;
  canvas.height = Math.min(maxHeight, height) * dpr;

  // Get context and scale it
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Scale all drawing operations by the dpr
  ctx.scale(dpr, dpr);

  // Clear with the scaled dimensions
  ctx.clearRect(0, 0, Math.min(maxWidth, width), Math.min(maxHeight, height));

  if (step < steps.length - 1) {
    const state: any = steps[step + 1];
    const last_state: any = steps[step];
    const delta_reward = state[0].observation.reward - last_state[0].observation.reward;

    const p1_move = state[1].observation.lastOpponentAction;
    const p2_move = state[0].observation.lastOpponentAction;

    const info = replay.info;
    const player1_text = info?.TeamNames?.[0] || 'Player 1';
    const player2_text = info?.TeamNames?.[1] || 'Player 2';

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const padding = 20;
    const row_width = (Math.min(maxWidth, width) - padding * 2) / 3;
    const label_x = padding;
    const player1_x = padding + row_width;
    const player2_x = padding + 2 * row_width;
    const middle_x = padding + row_width * 1.5;

    ctx.font = '30px sans-serif';
    ctx.fillStyle = '#FFFFFF';

    // Player Row
    ctx.fillText(player1_text, player1_x, label_y);
    ctx.fillText(player2_text, player2_x, label_y);

    // Action Id Row
    ctx.fillText('Action:', label_x, sign_id_y);
    ctx.fillText(p1_move, player1_x, sign_id_y);
    ctx.fillText(p2_move, player2_x, sign_id_y);

    // Action Name Row
    ctx.fillText('Name:', label_x, sign_name_y);
    ctx.fillText(SIGN_NAMES[p1_move], player1_x, sign_name_y);
    ctx.fillText(SIGN_NAMES[p2_move], player2_x, sign_name_y);

    // Emoji Row
    ctx.fillText('Icon:', label_x, sign_icon_y);
    ctx.fillText(SIGN_ICONS[p1_move], player1_x, sign_icon_y);
    ctx.fillText(SIGN_ICONS[p2_move], player2_x, sign_icon_y);

    // Result Row
    ctx.fillText('Result:', label_x, result_y);
    if (delta_reward === 1) {
      ctx.fillText('Win', player1_x, result_y);
    } else if (delta_reward === -1) {
      ctx.fillText('Win', player2_x, result_y);
    } else {
      ctx.fillText('Tie', middle_x, result_y);
    }

    // Reward Row
    ctx.fillText('Reward:', label_x, score_y);
    ctx.fillText(state[0].observation.reward, player1_x, score_y);
    ctx.fillText(state[1].observation.reward, player2_x, score_y);
  }
}
