import { RendererOptions } from '@kaggle-environments/core';

const ENERGY_TEXT = 'CGRWLPFDM A';
const WIDTH = 750;
const HEIGHT = 700;

const posY = (index: number, len: number) => {
  const center = 290;
  let height;
  if (len <= 8) {
    height = 35 * len;
  } else {
    height = 280;
  }
  return center + (height * (2 * index + 1 - len)) / len;
};

export function renderer(options: RendererOptions) {
  const { replay, step, agents } = options;
  const steps = replay.steps;
  const visList = (steps as any)[0][0].visualize;
  const playerNames = replay.info?.TeamNames || agents.map((a: any) => a?.name) || [];
  const players = [playerNames[0] || 'Player 0', playerNames[1] || 'Player 1'];

  let canvas = options.parent.querySelector('canvas');

  if (!canvas) {
    const container = document.createElement('div');

    // Style the container to be centered in its parent
    container.style.width = `${WIDTH}px`;
    container.style.height = `${HEIGHT}px`;
    container.style.margin = '0 auto'; // Center horizontally
    container.style.position = 'relative';
    options.parent.appendChild(container);

    canvas = document.createElement('canvas');

    // The following block ensures the canvas renders without fuzzy text
    const pixelRatio = window.devicePixelRatio || 1;

    // Set actual size in memory
    canvas.width = WIDTH * pixelRatio;
    canvas.height = HEIGHT * pixelRatio;

    // Set displayed size
    canvas.style.width = `${WIDTH}px`;
    canvas.style.height = `${HEIGHT}px`;

    container.appendChild(canvas);

    // Get context and scale it
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.scale(pixelRatio, pixelRatio);
    }

    if (visList) {
      for (let k = 0; k < 2; k++) {
        const button = document.createElement('button');
        button.style.width = '130px';
        button.style.height = '50px';
        button.style.left = k == 0 ? '230px' : '380px';
        button.style.top = '55px';
        button.style.position = 'absolute';
        button.style.zIndex = '1';
        button.innerHTML = 'Open Visualizer<br>' + players[k];
        button.addEventListener('click', () => {
          for (let i = 0; i < visList.length; i++) {
            for (let j = 0; j < 2; j++) {
              visList[i].current.players[j].ramainingTime = (steps as any)[i][j].observation.remainingOverageTime;
            }
          }
          visList[0].ps = players;

          const input = document.createElement('input');
          input.type = 'hidden';
          input.name = 'json';
          input.value = JSON.stringify(visList);

          const form = document.createElement('form');
          form.method = 'POST';
          form.action = 'https://ptcgvis.heroz.jp/Visualizer/Replay/';
          if (options?.replay?.info?.EpisodeId == null) {
            form.action += k;
          } else {
            form.action += options.replay.info.EpisodeId + '/' + k;
          }
          form.target = '_blank';
          form.appendChild(input);

          document.body.appendChild(form);
          form.submit();
        });
        container.appendChild(button);
      }
    } else {
      const ctx = canvas.getContext('2d');

      if (ctx) {
        ctx.strokeStyle = '#ccc';
        ctx.fillStyle = '#fff';
        ctx.font = '30px sans-serif';
        ctx.fillText('No visualizer data.', 10, 100);
        const error = (steps as any)[0][0].error;
        if (error) {
          ctx.fillText(error, 10, 150);
        }
      }
    }
  }

  if (visList.length <= step) {
    return;
  }
  const vis = visList[step];
  const state = vis.current;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#ccc';
  ctx.fillStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.font = '20px sans-serif';

  // Label for result at the end of the game
  if (state.result >= 0) {
    if (state.result == 2) {
      ctx.fillText('Draw', 330, 125);
    } else {
      ctx.fillText(players[state.result] + ' Win', 290, 140);
    }
  }

  ctx.font = '12px sans-serif';

  const drawCard = (x: number, y: number, card: any) => {
    ctx.beginPath();
    ctx.rect(x, y, 80, 60);
    ctx.stroke();
    let nm = card.name;
    let nm2 = null;
    if (nm.length >= 13) {
      for (let i = 0; i < nm.length; i++) {
        if (nm[i] == ' ') {
          nm2 = nm.substring(i + 1);
          nm = nm.substring(0, i);
          break;
        }
      }
    }
    ctx.fillText(nm, x + 5, y + 13);
    if (nm2 != null) {
      ctx.fillText(nm2, x + 5, y + 27);
    }
  };

  const drawField = (x: number, y: number, card: any) => {
    drawCard(x, y, card);
    ctx.fillText('HP ' + card.hp, x + 5, y + 41);
    let energy = '';
    for (const e of card.energies) {
      energy = energy + ENERGY_TEXT[e];
    }
    ctx.fillText(energy, x + 5, y + 55);
  };

  for (let j = 0; j < state.stadium.length; j++) {
    drawCard(330, 420, state.stadium[j]);
  }

  for (let i = 0; i < 2; i++) {
    const playerState = state.players[i];

    ctx.fillText('Active', i == 0 ? 245 : 425, 270);
    ctx.fillText('Bench', i == 0 ? 145 : 525, 10);
    ctx.fillText('Hand', i == 0 ? 15 : 655, 10);
    ctx.fillText('Deck ' + playerState.deckCount, i == 0 ? 258 : 438, 165);
    ctx.fillText('Discard ' + playerState.discard.length, i == 0 ? 245 : 425, 185);
    ctx.fillText('Prize ' + playerState.prize.length, i == 0 ? 258 : 438, 220);

    for (let j = 0; j < playerState.active.length; j++) {
      drawField(i == 0 ? 240 : 420, posY(j, playerState.active.length), playerState.active[j]);
    }
    for (let j = 0; j < playerState.bench.length; j++) {
      drawField(i == 0 ? 140 : 520, posY(j, playerState.bench.length), playerState.bench[j]);
    }
    for (let j = 0; j < playerState.hand.length; j++) {
      drawCard(i == 0 ? 10 : 650, posY(j, playerState.hand.length), playerState.hand[j]);
    }
  }
}
