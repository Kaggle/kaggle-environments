async function renderer(context) {
  const {
    act,
    agents,
    environment,
    frame,
    height = 400,
    interactive,
    isInteractive,
    parent,
    step,
    update,
    width = 400,
  } = context;

  // Canvas Setup.
  let canvas = parent.querySelector('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    parent.appendChild(canvas);
  }

  let c = canvas.getContext('2d');
  const size = Math.min(width, height);
  canvas.width = size;
  canvas.height = size;
  c.clearRect(0, 0, canvas.width, canvas.height);

  // Scale from 100x100 game space to canvas size
  const scale = size / 100.0;

  function drawCircle(x, y, r, color, fill = false) {
    c.beginPath();
    c.arc(x * scale, y * scale, r * scale, 0, 2 * Math.PI);
    c.fillStyle = color;
    c.strokeStyle = color;
    c.lineWidth = 2;
    if (fill) {
      c.fill();
    } else {
      c.stroke();
    }
  }

  function drawText(text, x, y, color, size_pt = 12) {
    c.font = `${(size_pt * scale) / 4}px sans-serif`;
    c.fillStyle = color;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(text, x * scale, y * scale);
  }

  // Draw Background (space!)
  c.fillStyle = '#000000';
  c.fillRect(0, 0, canvas.width, canvas.height);

  // Draw Sun
  const sunX = 50 * scale;
  const sunY = 50 * scale;
  const sunR = 10 * scale;
  // Glow effect
  const glow = c.createRadialGradient(sunX, sunY, sunR * 0.5, sunX, sunY, sunR * 2.5);
  glow.addColorStop(0, 'rgba(255, 200, 50, 0.6)');
  glow.addColorStop(0.5, 'rgba(255, 150, 20, 0.2)');
  glow.addColorStop(1, 'rgba(255, 100, 0, 0)');
  c.fillStyle = glow;
  c.fillRect(0, 0, canvas.width, canvas.height);
  // Sun body
  drawCircle(50, 50, 10, '#FFB800', true);
  drawCircle(50, 50, 10, '#FFD700', false);

  const current_step = environment.steps[step];
  const obs = current_step[0].observation;

  // Wong palette — colorblind-safe (blue, orange, teal, yellow, neutral grey)
  const colors = ['#0072B2', '#E69F00', '#009E73', '#F0E442', '#888888'];

  // Draw Comet tails (before planets so tails render behind)
  const cometPidSet = new Set(obs.comet_planet_ids || []);
  if (obs.comets) {
    obs.comets.forEach((group) => {
      const idx = group.path_index;
      group.planet_ids.forEach((pid, i) => {
        const path = group.paths[i];
        const tailLen = Math.min(idx + 1, path.length, 3);
        if (tailLen < 2) return;
        for (let t = 1; t < tailLen; t++) {
          const pi = idx - t;
          if (pi < 0) break;
          const alpha = 0.5 * (1 - t / tailLen);
          const x0 = path[pi][0] * scale;
          const y0 = path[pi][1] * scale;
          const x1 = path[pi + 1][0] * scale;
          const y1 = path[pi + 1][1] * scale;
          const width = (3 - (2 * t) / tailLen) * scale;
          c.beginPath();
          c.moveTo(x1, y1);
          c.lineTo(x0, y0);
          c.strokeStyle = `rgba(200, 220, 255, ${alpha})`;
          c.lineWidth = width;
          c.lineCap = 'round';
          c.stroke();
        }
      });
    });
  }

  // Draw Planets
  if (obs.planets) {
    obs.planets.forEach((p) => {
      const id = p[0];
      const owner = p[1];
      const x = p[2];
      const y = p[3];
      const r = p[4];
      const ships = p[5];

      const color = owner === -1 ? colors[4] : colors[owner];

      // Draw planet body (solid fill)
      drawCircle(x, y, r, color, true);
      // Comet border
      if (cometPidSet.has(id)) {
        drawCircle(x, y, r, '#FFFFFF', false);
      }

      // Draw ship count
      drawText(Math.floor(ships).toString(), x, y, '#FFFFFF', 12);
    });
  }

  // Draw Fleets as chevrons pointed in direction of travel
  if (obs.fleets) {
    obs.fleets.forEach((f) => {
      const owner = f[1];
      const x = f[2] * scale;
      const y = f[3] * scale;
      const angle = f[4];
      const ships = f[6];

      const color = colors[owner];
      // Scale chevron size by ship count: log scale, 1 ship = 0.5, 1000 = 3.0
      const sz = (0.5 + (2.5 * Math.log(ships)) / Math.log(1000)) * scale;

      c.save();
      c.translate(x, y);
      c.rotate(angle);

      // Standard chevron shape for all players
      c.beginPath();
      c.moveTo(sz, 0); // tip
      c.lineTo(-sz, -sz * 0.7); // top wing
      c.lineTo(-sz * 0.3, 0); // inner notch
      c.lineTo(-sz, sz * 0.7); // bottom wing
      c.closePath();
      c.fillStyle = color;
      c.fill();

      // Per-player marking lines for colorblind accessibility
      // P0: none, P1: 1 center line, P2: 2 lines (tip-to-wings), P3: 3 lines
      c.strokeStyle = 'rgba(255, 255, 255, 0.55)';
      c.lineWidth = sz * 0.15;
      c.lineCap = 'round';
      if (owner === 1 || owner === 3) {
        // Center line (tip to notch)
        c.beginPath();
        c.moveTo(sz * 0.8, 0);
        c.lineTo(-sz * 0.2, 0);
        c.stroke();
      }
      if (owner === 2 || owner === 3) {
        // Top line (tip toward top wing)
        c.beginPath();
        c.moveTo(sz * 0.6, -sz * 0.15);
        c.lineTo(-sz * 0.7, -sz * 0.5);
        c.stroke();
        // Bottom line (tip toward bottom wing)
        c.beginPath();
        c.moveTo(sz * 0.6, sz * 0.15);
        c.lineTo(-sz * 0.7, sz * 0.5);
        c.stroke();
      }

      c.restore();

      // Draw ship count: north side if fleet is in south half, south side if north
      const labelOffset = f[3] >= 50 ? -3 : 3;
      drawText(ships.toString(), f[2], f[3] + labelOffset, color, 8);
    });
  }

  // Draw Step Info
  c.fillStyle = '#FFFFFF';
  c.font = '16px sans-serif';
  c.textAlign = 'left';
  c.textBaseline = 'top';
  c.fillText(`Step: ${step}`, 10, 10);
}
