async function renderer({
  parent,
  step,
  frame,
  environment,
  width = 400,
  height = 400,
}) {
  // Configuration.
  const { size } = environment.configuration;
  const directions = ["NORTH", "EAST", "SOUTH", "WEST"];
  const state = environment.steps[step];
  const { halite, players } = state[0].observation;

  const colors = {
    bg: "#000B49",
    bgGradient: "#000B2A",
    players: ["#E2CD13", "#F24E4E", "#9BE38E", "#20BDFF"],
    ships: [
      [
        "#F1E61D",
        "#E2CD13",
        "#C0AE10",
        "#AA990E",
        "#716609",
        "#605708",
        "#161616",
      ],
      [
        "#F97575",
        "#F24E4E",
        "#CE4242",
        "#B63B3B",
        "#792727",
        "#672121",
        "#161616",
      ],
      [
        "#51E32A",
        "#36C61C",
        "#2EA818",
        "#299515",
        "#1B630E",
        "#17540C",
        "#161616",
      ],
      [
        "#2FDEFF",
        "#20BDFF",
        "#1AA1D9",
        "#168EBE",
        "#105F80",
        "#0F516D",
        "#161616",
      ],
    ],
  };

  // Rectangle coordinates on a 20x20 grid.
  const rects = {
    ship: [
      "9.5,0;9,1;8,3,1,2;7,5,1,2;6,7,1,2;7,10,2;9,9",
      "9,2,2,7;8,5,4,2;7,7,6,2;6,9,1,3;13,9,1,3;10,9;9,10,4,3;7,11,2,2;9,13,2",
      "10,1;11,3,1,2;12,5,1,2;13,7,1,2;8,13;11,13;9,14,2;9.5,15",
      "5,9;4,10;3,11;2,12;1,13",
      "5,10;4,11,2;3,12,3;14,10;14,11,2;14,12,3",
      "2,13,3;14,9;15,10;16,11;17,12;15,13,4",
      "9.5,6;9,7,2;8,8,4;7,9,2;11,9,2;6,12;5,13,3;4,14,5;13,12;12,13,3;11,14,5",
    ],
    flame: [
      "#FF972E;5,15,3,1,4;6,16,1,1,4;4.5,15,4,1,4,0.33;5,16,3,1,4,0.33;6,17,1,1,4,0.33;4,15,5,2,4,0.66;5,17,3,1,4,0.66;6,18,1,1,4,0.66",
      "#FEF545;6,15,1,1,4;5.5,15,2,1,4,0.33;6,16,1,1,4,0.33;5,15,3,1,4,0.66;5.5,16,2,1,4,0.66;6,17,1,1,4,0.66",
      "#FFF5FF;6,15,1,1,4,0.66",
    ],
    shipyard: [
      "9,9,2,2;2,2,3,3,15;2,4,5,1,15;3,5,5,1,15;4,6,5,1,15",
      "#FFFFFF33;9,9,2,2,0,0.2;2,2,2,2,14;2,3,3,1,15;3,4,3,1,15;4,5,3,1,15;5,6,3,1,15;",
      "#FFFFFF66;9,9,2,2,0,0.4;6,6,1,1,14,0.6;5,5,1,1,14,0.7;4,4,1,1,14,0.8;3,3,1,1,14,0.9;2,2,1,1,14,1",
    ],
    explosion: [
      "#C84302;7,7,1,1,14;6,9,1,2,4;5,5,1,1,14,0.25;9,5,2,10,0,0.25;6,9,8,2,0,0.25;3,3,1,1,14,0.5;7,6,1,1,15,0.5;5,4,1,1,14,0.75;4,5,1,1,14,0.75;8,5,1,1,14,0.75;9,4,2,1,2,0.75;4,8,1,1,14,0.75;7,2,1,1,14,0.75;",
      "#FF972E;9,6,2,1,2;8,9,4,2;4,9,1,2,4,0.25,0.74;9,6,2,8,0,0.25;7,7,6,2,2,0.25;2,9,1,2,4,0.5;6,7,8,2,2,0.75;9,5,2,10,0,0.75;8,6,4,8,0,0.5;6,6,1,1,14,0.75;5,5,1,1,14,0.75;5,7,1,1,14,0.75;7,4,1,1,14,0.75;",
      "#FEF545;9,8,2,4;9,7,2,6,0,0.25;8,8,4,4,0,0.25;8,7,1,1,14,0.5;8,7,4,6,1,0.75;9,6,2,1,2,0.75;",
      "#FFF5FF;9,9,2,2;8,9,4,2,1,0.25;7,9,6,2,0,0.5;9,7,2,6,0,0.75",
    ],
    largeHalite: [
      "#008DFF;17,6;2,13;9,1,2,18,1;5,7,10,6,1",
      "#00C9FF;9,3,2,14;3,9,14,2",
      "#00FFFF;6,2;13,17;4,9,12,2,1;7,7,6,6",
      "#FFFFFF;13,2;17,13;2,6;6,17;6,9,8,2,1",
    ],
    mediumHalite: [
      "#008DFF;6,4;16,7;16,12;6,15;4,9,12,2,1;6,8,8,4,1",
      "#00C9FF;9,5,2,10,1;",
      "#00FFFF;13,4;3,7;3,12;13,15;9,6,2,8,1;8,8,4,4",
      "#FFFFFF;9,7,2,6,1",
    ],
    smallHalite: [
      "#008DFF;13.5,6.5;13.5,12.5;9.5,5.5,1,9,1;8.5,6.5,3,7,1",
      "#00C9FF;9.5,6.5,1,7,1",
      "#00FFFF;5.5,6.5;5.5,12.5;9.5,7.5,1,5,1;8.5,8.5,3,3",
      "#FFFFFF;9.5,8.5,1,3,1",
    ],
  };

  // Helper Functions.
  const createElement = (type, id) => {
    const el = document.createElement(type);
    el.id = id;
    parent.appendChild(el);
    return el;
  };

  const getCanvas = (id, options = { clear: false, alpha: false }) => {
    let canvas = document.querySelector(`#${id}`);
    if (!canvas) {
      canvas = createElement("canvas", id);
      canvas.width = options.width || width;
      canvas.height = options.height || height;
      canvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%; 
      `;
    }
    const ctx = canvas.getContext("2d", { alpha: options.alpha });
    if (options.clear) ctx.clearRect(0, 0, canvas.width, canvas.height);
    return [canvas, ctx];
  };

  const data = function(selector, key, value) {
    const el =
      typeof selector === "string"
        ? document.querySelector(selector)
        : selector;
    if (arguments.length === 3) {
      el.setAttribute(`data-${key}`, JSON.stringify(value));
      return value;
    }
    if (el.hasAttribute(`data-${key}`)) {
      return JSON.parse(el.getAttribute(`data-${key}`));
    }
    return null;
  };

  const move = (ctx, options = {}, fn) => {
    const { x, y, width, height, angle, scale } = {
      x: 0,
      y: 0,
      width: 100,
      height: 100,
      angle: 0,
      ...options,
    };
    ctx.save();
    ctx.translate(x, y);
    if (scale) ctx.scale(scale, scale);
    if (angle) {
      ctx.translate(width / 2, height / 2);
      ctx.rotate((Math.PI * angle) / 180);
      ctx.translate(-width / 2, -height / 2);
    }
    fn();
    ctx.restore();
  };

  const drawRects = (
    ctx,
    rects,
    color,
    scale = 1,
    gridSize = 20,
    drawFrame = -1
  ) => {
    if (drawFrame == -1) drawFrame = frame;
    // rects="x,y,w,h,specials,minFrame,maxFrame;..."
    ctx.save();
    if (color) ctx.fillStyle = color;
    ctx.beginPath();
    const drawSpecials = (x, y, w, h, special) => {
      const size = gridSize * scale;
      if ((special & 1) === 1) ctx.rect(y, x, h, w); // swap x/y and w/h
      if ((special & 2) === 2) ctx.rect(x, size - y - h, w, h); // Mirror over X Axis
      if ((special & 4) === 4) ctx.rect(size - x - w, y, w, h); // Mirror over Y Axis
      if ((special & 8) === 8) ctx.rect(size - x - w, size - y - h, w, h); // Mirror over X & Y Axis
      // Repeat mirroring if a swap occurred.
      if ((special & 1) === 1) drawSpecials(y, x, h, w, special - 1);
    };
    rects
      .replace(/\s/g, "")
      .split(";")
      .filter(r => !!r)
      .forEach(coords => {
        // Apply a fill style.
        if (coords[0] == "#" || coords[0] == "r") {
          ctx.fillStyle = coords;
          return;
        }
        const defaultCoords = ["0", "0", "1", "1", "0", "0", "1"];
        coords = coords.split(",");
        let [x, y, w, h, special, minFrame, maxFrame] = defaultCoords.map(
          (v, i) =>
            parseFloat(coords.length > i ? coords[i] : v) * (i < 4 ? scale : 1)
        );
        if (minFrame > drawFrame || maxFrame < drawFrame) return;
        ctx.rect(x, y, w, h);
        drawSpecials(x, y, w, h, special);
      });
    ctx.fill();
    ctx.closePath();
    ctx.restore();
  };

  const getColRow = pos => [pos % size, Math.floor(pos / size)];

  const getMovePos = (pos, direction) => {
    const [col, row] = getColRow(pos);
    switch (direction) {
      case "NORTH":
        return pos >= size ? pos - size : Math.pow(size, 2) - size + col;
      case "SOUTH":
        return pos + size >= Math.pow(size, 2) ? col : pos + size;
      case "EAST":
        return col < size - 1 ? pos + 1 : row * size;
      case "WEST":
        return col > 0 ? pos - 1 : (row + 1) * size - 1;
      default:
        throw new Error(`"${direction}" is not a valid move action.`);
    }
  };

  const getCoords = pos => {
    const [col, row] = getColRow(pos);
    return {
      col,
      row,
      scale: cellScale * cellInset,
      dx: xOffset + cellSize * col + (cellSize - cellSize * cellInset) / 2,
      dy: yOffset + cellSize * row + (cellSize - cellSize * cellInset) / 2,
      ds: cellScale * cellInset * fixedCellSize,
      ss: fixedCellSize,
    };
  };

  const getShipDir = (playerIndex, uid) => {
    for (let s = step; s >= 0; s--) {
      const action = environment.steps[s][playerIndex].action;
      if (uid in action) return Math.max(directions.indexOf(action[uid]), 0);
    }
    for (let s = step + 1; s < environment.steps.length; s++) {
      const action = environment.steps[s][playerIndex].action;
      if (uid in action) return Math.max(directions.indexOf(action[uid]), 0);
    }
    return 0;
  };

  // First time setup.
  if (!parent.querySelector("#buffer")) {
    const [bufferCanvas, ctx] = getCanvas("buffer", {
      alpha: true,
      clear: false,
      width: 900,
      height: 700,
    });

    // Setup common fields.
    const cellInset = 0.8;
    const fixedCellSize = 100;
    const minOffset = Math.min(height, width) > 400 ? 30 : 4;
    const cellSize = Math.min(
      (width - minOffset * 2) / size,
      (height - minOffset * 2) / size
    );
    const haliteRotations = Array(size * size)
      .fill(0)
      .map(_ => Math.random() * 360);

    data(bufferCanvas, "storage", {
      cellInset,
      cellScale: cellSize / fixedCellSize,
      cellSize,
      fixedCellSize,
      haliteRotations,
      maxCellHalite: Math.max(...halite),
      xOffset: Math.max(0, (width - cellSize * size) / 2),
      yOffset: Math.max(0, (height - cellSize * size) / 2),
    });

    // Pre-render visualizations (100x100 cells).
    // Halite
    ["largeHalite", "mediumHalite", "smallHalite"].forEach((rectsName, i) => {
      move(ctx, { x: 0, y: 100 * i }, () => {
        rects[rectsName].forEach(v => drawRects(ctx, v, null, 5));
      });
    });
    // Explosions.
    for (let s = 0; s < 4; s++) {
      move(ctx, { x: 100, y: 100 * s }, () => {
        rects.explosion.forEach(v => drawRects(ctx, v, null, 5, 20, 1 - s / 3));
      });
    }
    // Flames.
    for (let s = 0; s < 3; s++) {
      for (let d in directions) {
        move(ctx, { x: 200 + s * 100, y: 100 * d, angle: d * 90 }, () => {
          rects.flame.forEach(v => drawRects(ctx, v, null, 5, 20, s / 3));
        });
      }
    }
    // Ships.
    colors.ships.forEach((color, n) => {
      for (let d in directions) {
        move(ctx, { x: 500 + 100 * n, y: d * 100, angle: d * 90 }, () => {
          rects.ship.forEach((v, i) => drawRects(ctx, v, color[i], 5));
        });
      }
    });
  }

  // Restore Canvases.
  const [bufferCanvas] = getCanvas("buffer", {
    alpha: true,
    clear: false,
  });
  const [bgCanvas, bgCtx] = getCanvas("background", {
    alpha: true,
    clear: false,
  });
  const [, fgCtx] = getCanvas("foreground", {
    alpha: true,
    clear: true,
  });

  // Expand storage.
  const {
    cellInset,
    cellScale,
    cellSize,
    fixedCellSize,
    haliteRotations,
    maxCellHalite,
    xOffset,
    yOffset,
  } = data(bufferCanvas, "storage");

  // Render Background once per step (Gradient + Halite)
  if (data(bgCanvas, "step") !== step) {
    data(bgCanvas, "step", step);
    bgCtx.fillStyle = colors.bg;
    bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);

    const r = Math.min(height, width) / 2;
    const bgStyle = bgCtx.createRadialGradient(r, r, 0, r, r, r);
    bgStyle.addColorStop(0, colors.bg);
    bgStyle.addColorStop(1, colors.bgGradient);
    bgCtx.fillStyle = bgStyle;
    bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);

    // Render the halite.
    halite.forEach((cellHalite, pos) => {
      if (!cellHalite) return;

      let { dx, dy, ds, ss } = getCoords(pos);
      let sx = 0;
      let sy = 0;

      const pct = Math.min(cellHalite, maxCellHalite) / maxCellHalite;
      let scale = 1;

      // Scale by the halite size.
      if (pct > 0.7) {
        scale = pct;
      } else if (pct < 0.3) {
        scale = pct / 0.3;
        sy = 200;
      } else {
        sy = 100;
        scale = pct + 0.3;
      }

      // Apply the scale.
      scale = Math.max(0.3, scale);
      dx += (ds - ds * scale) / 2;
      dy += (ds - ds * scale) / 2;
      ds *= scale;

      // Rotate the halite to get a bit of randomness.
      move(
        bgCtx,
        { x: dx, y: dy, width: ds, height: ds, angle: haliteRotations[pos] },
        () => {
          bgCtx.drawImage(bufferCanvas, sx, sy, ss, ss, 0, 0, ds, ds);
        }
      );
    });
  }

  // Render Foreground (every frame).

  // Draw Shipyards.
  players.forEach((player, player_index) => {
    Object.values(player[1]).forEach(pos => {
      const { scale, dx, dy } = getCoords(pos);
      move(fgCtx, { x: dx, y: dy, scale }, () => {
        rects.shipyard.forEach((v, i) =>
          drawRects(fgCtx, v, colors.players[player_index], 5)
        );
      });
    });
  });

  // Draw collisions.
  if (step > 0) {
    const board = Array(size * size)
      .fill(0)
      .map(() => ({ shipyard: -1, ship: null, collision: false }));
    players.forEach((player, player_index) => {
      const [, shipyards, ships] = player;
      Object.values(shipyards).forEach(
        pos => (board[pos].shipyard = player_index)
      );
      Object.entries(ships).forEach(([uid, [pos]]) => (board[pos].ship = uid));
    });
    environment.steps[step - 1][0].observation.players.forEach(
      (player, player_index) => {
        const status = state[player_index].status;
        if (status != "ACTIVE" && status != "DONE") return;
        const [, shipyards, ships] = player;
        const action = environment.steps[step][player_index].action;
        // Stationary ships collecting Halite.
        Object.entries(ships).forEach(([uid, [pos]]) => {
          if (uid in action) return;
          if (board[pos].ship !== uid) board[pos].collision = true;
        });
        // Convert to shipyard, Spawn ship, or Move ship.
        Object.entries(action).forEach(([uid, value]) => {
          if (value === "SPAWN") {
            if (
              !board[shipyards[uid]].ship ||
              parseInt(board[shipyards[uid]].ship.split("-")[0]) !== step
            ) {
              board[shipyards[uid]].collision = true;
            }
          } else if (value !== "CONVERT") {
            const toPos = getMovePos(ships[uid][0], value);
            if (board[toPos].ship !== uid) board[toPos].collision = true;
          }
        });
      }
    );

    board.forEach(({ collision }, pos) => {
      if (!collision) return;
      const { dx, dy, ds, ss } = getCoords(pos);
      const sx = 100;
      const sy = 100 * Math.round(4 - frame * 4);
      fgCtx.drawImage(bufferCanvas, sx, sy, ss, ss, dx, dy, ds, ds);
    });
  }

  // Draw Ships.
  players.forEach((player, player_index) => {
    Object.entries(player[2]).forEach(([uid, [pos]]) => {
      const shipx = 500 + 100 * player_index;
      const flamex = 200 + (frame <= 0.33 ? 0 : frame >= 0.66 ? 200 : 100);
      const { dx, dy, ds } = getCoords(pos);
      const sy = getShipDir(player_index, uid) * 100;
      const ss = fixedCellSize;
      fgCtx.drawImage(bufferCanvas, shipx, sy, ss, ss, dx, dy, ds, ds);
      fgCtx.drawImage(bufferCanvas, flamex, sy, ss, ss, dx, dy, ds, ds);
    });
  });
}
