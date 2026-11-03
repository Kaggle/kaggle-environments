// Replay renderer for the Weeping Angel Arena.
//
// Draws a row of file cells per step: empty files are grey, files hosting a
// lurking Angel are amber, caught Angels green, corrupted red. Files blue
// covered this step get a blue outline. The header shows the step and score.
//
// NOTE: written to the kaggle-environments renderer(context) contract but not
// yet verified in a live browser player -- check before submitting upstream.
function renderer(context) {
  const { parent, step, environment } = context;
  const width = context.width || 700;
  const height = context.height || 180;

  let canvas = parent.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    parent.appendChild(canvas);
  }
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);

  const steps = environment.steps;
  const frame = steps[Math.min(step, steps.length - 1)];
  const blue = frame[0];
  const red = frame[1];
  const angels = (red.observation && red.observation.angels) || [];
  const config = environment.configuration || {};
  const nFiles = config.nFiles || 12;
  const cover = new Set((blue.action || []).map((x) => parseInt(x, 10)));

  const statusByFile = {};
  for (const a of angels) statusByFile[a[1]] = a[2]; // 0 lurk, 1 caught, 2 corrupt
  const COLORS = { empty: "#e5e7eb", lurk: "#f59e0b", caught: "#27ae60", corrupt: "#c0392b" };

  ctx.fillStyle = "#333";
  ctx.font = "14px monospace";
  const caught = angels.filter((a) => a[2] === 1).length;
  const corrupted = angels.filter((a) => a[2] === 2).length;
  ctx.fillText(
    `step ${blue.observation.step}   blue ${blue.reward || 0} / red ${red.reward || 0}` +
      `   caught ${caught}  corrupted ${corrupted}`,
    16,
    26
  );

  const pad = 16;
  const cellW = (width - pad * 2) / nFiles;
  const cellH = 60;
  const top = 60;
  for (let f = 0; f < nFiles; f++) {
    const s = statusByFile[f];
    let color = COLORS.empty;
    if (s === 0) color = COLORS.lurk;
    else if (s === 1) color = COLORS.caught;
    else if (s === 2) color = COLORS.corrupt;
    const x = pad + f * cellW;
    ctx.fillStyle = color;
    ctx.fillRect(x + 2, top, cellW - 4, cellH);
    if (cover.has(f)) {
      ctx.strokeStyle = "#2980b9";
      ctx.lineWidth = 3;
      ctx.strokeRect(x + 2, top, cellW - 4, cellH);
    }
    ctx.fillStyle = "#444";
    ctx.font = "11px monospace";
    ctx.fillText(String(f), x + cellW / 2 - 4, top + cellH + 16);
  }
}
