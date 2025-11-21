import { PIECE_IMAGES, COLORS } from './consts';
import { LegacyRendererOptions } from '@kaggle-environments/core';
import { data, getSpawnValue } from './utils';

const directions = ['NORTH', 'EAST', 'SOUTH', 'WEST'];

interface BoardPiece {
  shipyard: number;
  ship: string | null;
  collision: boolean;
  shipPlayer: number | null;
}

interface MoveOptions {
  x: number;
  y: number;
  width?: number;
  height?: number;
  angle?: number;
  scale?: number;
}

export function renderer(options: LegacyRendererOptions) {
  console.log(options);
  const { step, parent, replay, width = 400, height = 400 } = options;
  const frame = 0;

  // Configuration.
  const { size } = replay.configuration;
  const steps = replay.steps as any[];
  const agents = replay.info?.agents;
  const state: any = steps[step];
  const { kore, players } = state[0].observation;

  // Helper Functions.
  const createElement = (type: any, id: string) => {
    const el = document.createElement(type);
    el.id = id;
    parent.appendChild(el);
    return el;
  };

  const getCanvas = (id: string, options = { clear: false, alpha: false }) => {
    let canvas: any = document.querySelector(`#${id}`);
    if (!canvas) {
      canvas = createElement('canvas', id);
      canvas.width = width;
      canvas.height = height;
      canvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%; 
      `;
    }
    const ctx = canvas.getContext('2d', { alpha: options.alpha });
    if (options.clear) ctx.clearRect(0, 0, canvas.width, canvas.height);
    return [canvas, ctx];
  };

  const move = (ctx: any, options: MoveOptions, fn?: () => void) => {
    const { x, y, width = 100, height = 100, angle, scale } = options;
    ctx.save();
    ctx.translate(x, y);
    if (scale) ctx.scale(scale, scale);
    if (angle) {
      ctx.translate(width / 2, height / 2);
      ctx.rotate((Math.PI * angle) / 180);
      ctx.translate(-width / 2, -height / 2);
    }
    if (fn) fn();
    ctx.restore();
  };

  const drawRects = (ctx: any, rects: string, color: string | null, scale = 1, gridSize = 20, drawFrame = -1) => {
    if (drawFrame == -1) drawFrame = frame;
    // rects="x,y,w,h,specials,minFrame,maxFrame;..."
    ctx.save();
    if (color) ctx.fillStyle = color;
    ctx.beginPath();
    const drawSpecials = (x: number, y: number, w: number, h: number, special: number) => {
      const size = gridSize * scale;
      if ((special & 1) === 1) ctx.rect(y, x, h, w); // swap x/y and w/h
      if ((special & 2) === 2) ctx.rect(x, size - y - h, w, h); // Mirror over X Axis
      if ((special & 4) === 4) ctx.rect(size - x - w, y, w, h); // Mirror over Y Axis
      if ((special & 8) === 8) ctx.rect(size - x - w, size - y - h, w, h); // Mirror over X & Y Axis
      // Repeat mirroring if a swap occurred.
      if ((special & 1) === 1) drawSpecials(y, x, h, w, special - 1);
    };
    rects
      .replace(/\s/g, '')
      .split(';')
      .filter((r) => !!r)
      .forEach((coords) => {
        // Apply a fill style.
        if (coords[0] == '#' || coords[0] == 'r') {
          ctx.fillStyle = coords;
          return;
        }
        const defaultCoords = ['0', '0', '1', '1', '0', '0', '1'];
        const [x, y, w, h, special, minFrame, maxFrame] = defaultCoords.map(
          (v, i) => parseFloat(coords.length > i ? coords[i] : v) * (i < 4 ? scale : 1)
        );
        if (minFrame > drawFrame || maxFrame < drawFrame) return;
        ctx.rect(x, y, w, h);
        drawSpecials(x, y, w, h, special);
      });
    ctx.fill();
    ctx.closePath();
    ctx.restore();
  };

  const getColRow = (pos: number) => [pos % size, Math.floor(pos / size)];

  const getDirStrFromIdx = (dirIdx: number) => {
    switch (dirIdx) {
      case 0:
        return 'NORTH';
      case 1:
        return 'EAST';
      case 2:
        return 'SOUTH';
      case 3:
        return 'WEST';
      default:
        throw new Error(`"${dirIdx}" is not a valid direction idx.`);
    }
  };

  const getDirStrFromChar = (dirChar: string) => {
    switch (dirChar) {
      case 'N':
        return 'NORTH';
      case 'E':
        return 'EAST';
      case 'S':
        return 'SOUTH';
      case 'W':
        return 'WEST';
      default:
        throw new Error(`"${dirChar}" is not a valid direction char.`);
    }
  };

  const getMovePos = (pos: number, direction: string) => {
    const [col, row] = getColRow(pos);
    switch (direction) {
      case 'NORTH':
        return pos >= size ? pos - size : Math.pow(size, 2) - size + col;
      case 'SOUTH':
        return pos + size >= Math.pow(size, 2) ? col : pos + size;
      case 'EAST':
        return col < size - 1 ? pos + 1 : row * size;
      case 'WEST':
        return col > 0 ? pos - 1 : (row + 1) * size - 1;
      default:
        throw new Error(`"${direction}" is not a valid move action.`);
    }
  };

  const getCoords = (pos: number) => {
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

  // First time setup.
  if (!parent.querySelector('#buffer')) {
    const [bufferCanvas, ctx] = getCanvas('buffer', {
      alpha: true,
      clear: false,
      // width: 900,
      // height: 700,
    });

    // Setup common fields.
    const cellInset = 0.8;
    const fixedCellSize = 100;
    const minOffset = Math.min(height, width) > 400 ? 30 : 4;
    const cellSize = Math.min((width - minOffset * 2) / size, (height - minOffset * 2) / size);
    const koreRotations = Array(size * size)
      .fill(0)
      .map(() => Math.random() * 360);

    data(bufferCanvas, 'storage', {
      cellInset,
      cellScale: cellSize / fixedCellSize,
      cellSize,
      fixedCellSize,
      koreRotations,
      maxCellKore: Math.max(...kore),
      xOffset: Math.max(0, (width - cellSize * size) / 2),
      yOffset: Math.max(0, (height - cellSize * size) / 2),
    });

    // Pre-render visualizations (100x100 cells).
    // Kore
    ['largeKore', 'mediumKore', 'smallKore'].forEach((rectsName, i) => {
      move(ctx, { x: 0, y: 100 * i }, () => {
        PIECE_IMAGES[rectsName].forEach((v: string) => drawRects(ctx, v, null, 5));
      });
    });
    // Explosions.
    for (let s = 0; s < 4; s++) {
      move(ctx, { x: 100, y: 100 * s }, () => {
        PIECE_IMAGES.explosion.forEach((v: string) => drawRects(ctx, v, null, 5, 20, 1 - s / 3));
      });
    }
    // Flames.
    for (let s = 0; s < 3; s++) {
      for (const d in directions) {
        move(ctx, { x: 200 + s * 100, y: 100 * directions.indexOf(d), angle: directions.indexOf(d) * 90 }, () => {
          PIECE_IMAGES.flame.forEach((v: string) => drawRects(ctx, v, null, 5, 20, s / 3));
        });
      }
    }
    // Ships.
    COLORS.ships.forEach((color: string, n: number) => {
      for (const d in directions) {
        move(ctx, { x: 500 + 100 * n, y: directions.indexOf(d) * 100, angle: directions.indexOf(d) * 90 }, () => {
          PIECE_IMAGES.ship.forEach((v: string, i: number) => drawRects(ctx, v, color[i], 5));
        });
      }
    });
    // Shipyards.
    COLORS.ships.forEach((color: string, n: number) => {
      move(ctx, { x: 500 + 100 * n, y: 400 }, () => {
        PIECE_IMAGES.shipyard.forEach((v: string, i: number) => drawRects(ctx, v, color[i], 5));
      });
    });
  }

  // Restore Canvases.
  const [bufferCanvas] = getCanvas('buffer', {
    alpha: true,
    clear: false,
  });
  const [bgCanvas, bgCtx] = getCanvas('background', {
    alpha: true,
    clear: false,
  });
  const [, fgCtx] = getCanvas('foreground', {
    alpha: true,
    clear: true,
  });

  // Expand storage.
  const { cellInset, cellScale, cellSize, fixedCellSize, koreRotations, xOffset, yOffset } = data(
    bufferCanvas,
    'storage'
  );

  const topLeftCell = getCoords(0);
  const botRightCell = getCoords(size * size - 1);

  const renderKore = (ctx: any, pos: number, kore: number, maxKore: number, scaleFactor: number, rotate: boolean) => {
    if (kore <= 0) return;

    const coords = getCoords(pos);
    let { dx, dy, ds } = coords;
    const { ss } = coords;

    const sx = 0;
    let sy = 0;

    const pct = Math.min(1, kore / maxKore);
    let scale = 1;

    // Scale by the kore size.
    if (pct > 0.7) {
      scale = pct;
    } else if (pct > 0.3) {
      sy = 100;
      scale = pct + 0.3;
    } else {
      sy = 200;
      scale = pct * 3;
    }

    // Apply the scale.
    scale = Math.max(0.3, scaleFactor * scale);
    dx += (ds - ds * scale) / 2;
    dy += (ds - ds * scale) / 2;
    ds *= scale;

    // Rotate the kore to get a bit of randomness, if desired.
    move(ctx, { x: dx, y: dy, width: ds, height: ds, angle: rotate ? koreRotations[pos] : 0 }, () =>
      ctx.drawImage(bufferCanvas, sx, sy, ss, ss, 0, 0, ds, ds)
    );
  };

  // Render Background once per step (Gradient + Kore)
  const boxPadding = height * 0.007;
  if (data(bgCanvas, 'step') !== step) {
    data(bgCanvas, 'step', step);
    bgCtx.fillStyle = COLORS.bg;
    bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);

    const r = Math.min(height, width) / 2;
    const bgStyle = bgCtx.createRadialGradient(r, r, 0, r, r, r);
    bgStyle.addColorStop(0, COLORS.bg);
    bgStyle.addColorStop(1, COLORS.bgGradient);
    bgCtx.fillStyle = bgStyle;
    bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);

    // Render bounding box.
    bgCtx.strokeStyle = 'white';
    bgCtx.lineWidth = 0.5;
    bgCtx.strokeRect(
      topLeftCell.dx - boxPadding,
      topLeftCell.dy - boxPadding,
      botRightCell.dx + botRightCell.ds - topLeftCell.dx + 2 * boxPadding,
      botRightCell.dy + botRightCell.ds - topLeftCell.dy + 2 * boxPadding
    );

    // Render the kore.
    kore.forEach((cellKore: number, pos: number) => renderKore(bgCtx, pos, cellKore, 500, 1, true));
  }

  // Render Foreground (every frame).

  // Draw Shipyards.
  const shipFontSizePx = Math.round(height / 64);
  fgCtx.fillStyle = '#FFFFFF';
  fgCtx.textBaseline = 'top';
  fgCtx.textAlign = 'left';
  players.forEach((player: any, playerIndex: number) => {
    Object.entries(player[1]).forEach(([, value]) => {
      if (Array.isArray(value) && value.length >= 3) {
        const pos = value[0];
        const shipCount = value[1];
        const turnsControlled = value[2];
        const shipx = 500 + 100 * playerIndex;
        const ss = fixedCellSize;
        const { dx, dy, ds } = getCoords(pos);
        fgCtx.drawImage(bufferCanvas, shipx, 400, ss, ss, dx, dy, ds, ds);
        fgCtx.font = `bold ${shipFontSizePx}px sans-serif`;
        fgCtx.fillText(shipCount, dx, dy);
        const spawnValue = getSpawnValue(turnsControlled);
        fgCtx.font = `bold ${Math.round(0.7 * shipFontSizePx)}px sans-serif`;
        fgCtx.fillText(spawnValue, dx + Math.round(0.85 * ds), dy + Math.round(0.85 * ds));
      }
    });
  });

  // Draw Ships and a smaller Kore icon according to their current cargo.
  players.forEach((player: any, playerIndex: number) => {
    Object.entries(player[2]).forEach(([, value]) => {
      if (Array.isArray(value) && value.length >= 4) {
        const pos = value[0];
        const cargo = value[1];
        const shipCount = value[2];
        const directionIdx = value[3];

        const shipx = 500 + 100 * playerIndex;
        const flamex = 200 + 100 * Math.min(2, Math.floor(3 * frame));
        const { dx, dy, ds } = getCoords(pos);
        const sy = directionIdx * 100;
        const ss = fixedCellSize;
        fgCtx.drawImage(bufferCanvas, shipx, sy, ss, ss, dx, dy, ds, ds);
        fgCtx.drawImage(bufferCanvas, flamex, sy, ss, ss, dx, dy, ds, ds);
        fgCtx.fillText(shipCount, dx, dy);
        renderKore(fgCtx, pos, cargo, 1500, 0.6, false);
      }
    });
  });

  // Draw collisions.
  if (step > 1) {
    const board: BoardPiece[] = Array(size * size)
      .fill(0)
      .map(() => ({ shipyard: -1, ship: null, collision: false, shipPlayer: null }));
    players.forEach((player: any, playerIndex: number) => {
      const [shipyards, ships] = player;
      // For shipyards
      Object.entries(shipyards).forEach(([, shipyardData]) => {
        if (Array.isArray(shipyardData) && shipyardData.length >= 2) {
          const pos = shipyardData[0];
          board[pos].shipyard = playerIndex;
        }
      });

      // For ships
      Object.entries(ships).forEach(([uid, shipData]) => {
        if (Array.isArray(shipData) && shipData.length >= 1) {
          const pos = shipData[0];
          board[pos].ship = uid;
          board[pos].shipPlayer = playerIndex;
        }
      });
    });

    steps[step - 1][0].observation.players.forEach((player: any, playerIndex: number) => {
      const [, , fleets] = player;

      Object.entries(fleets).forEach(([, fleetData]) => {
        // First, ensure fleetData is an array with the expected structure
        if (Array.isArray(fleetData) && fleetData.length >= 3) {
          const pos = fleetData[0];
          const directionIdx = fleetData[1];
          let flightPath = fleetData[2];

          let dir = getDirStrFromIdx(directionIdx);
          while (flightPath.length > 0 && flightPath[0] == '0') {
            flightPath = flightPath.substring(1);
          }
          if (flightPath.length > 0 && 'NESW'.includes(flightPath[0])) {
            dir = getDirStrFromChar(flightPath[0]);
          }
          const toPos = getMovePos(pos, dir);
          // if there is an enemy shipyard in the next square,
          const shipyardNextSquare = board[toPos].shipyard !== -1;
          const shipyardPreviousSquare = board[pos].shipyard !== -1;
          const enemyOrNoShipNextSquare = board[toPos].shipPlayer !== playerIndex;
          const enemyShipyardNextSquare = shipyardNextSquare && board[toPos].shipyard !== playerIndex;
          const alliedShipyardNextSquare = shipyardNextSquare && board[toPos].shipyard === playerIndex;
          const alliedShipyardPreviousSquare = shipyardPreviousSquare && board[pos].shipyard === playerIndex;
          if (alliedShipyardPreviousSquare || alliedShipyardNextSquare) {
            // don't explode
          } else if (enemyShipyardNextSquare) {
            board[toPos].collision = true;
          } else if (enemyOrNoShipNextSquare) {
            board[toPos].collision = true;
          }
        }
      });
    });

    board.forEach(({ collision }, pos) => {
      if (!collision) return;
      const { dx, dy, ds, ss } = getCoords(pos);
      const sx = 100;
      const sy = 100 * Math.round(4 * (1 - frame));
      fgCtx.drawImage(bufferCanvas, sx, sy, ss, ss, dx, dy, ds, ds);
    });
  }

  const scoreboardFontSizePx = Math.round(height / 36);
  const scoreboardPaddingPx = Math.max(1, scoreboardFontSizePx / 4);
  const scoreboardLineYDiffPx = scoreboardFontSizePx + scoreboardPaddingPx;

  const getKore = (player: any[]) => player[0];
  const getCargo = (player: any[]) =>
    Object.entries(player[2])
      .map(([, v]) => {
        // Check if v is an array with at least 2 elements
        if (Array.isArray(v) && v.length >= 2) {
          return v[1]; // Access the cargo value
        }
        return 0; // Default value if structure doesn't match
      })
      .reduce((a, b) => a + b, 0);
  const getShipCount = (player: any[]) =>
    Object.entries(player[2])
      .map(([, v]) => {
        // Check if v is an array with at least 2 elements
        if (Array.isArray(v) && v.length >= 2) {
          return v[2]; // Access the cargo value
        }
        return 0; // Default value if structure doesn't match
      })
      .reduce((a, b) => a + b, 0) +
    Object.entries(player[1])
      .map(([, v]) => {
        // Check if v is an array with at least 2 elements
        if (Array.isArray(v) && v.length >= 2) {
          return v[1]; // Access the cargo value
        }
        return 0; // Default value if structure doesn't match
      })
      .reduce((a, b) => a + b, 0);
  const getNumFleets = (player: any[]) => Object.entries(player[2]).length;
  const getNumShipyards = (player: any[]) => Object.entries(player[1]).length;

  // Writes two lines, "Kore" and "Cargo", and returns y value for what would be the third line.
  const writeScoreboardText = (ctx: any, player: any, x: number, y: number) => {
    ctx.fillText(`Kore: ${Math.floor(getKore(player))}`, x, y);
    ctx.fillText(`Cargo: ${Math.floor(getCargo(player))}`, x, y + scoreboardLineYDiffPx);
    ctx.fillText(`Ships: ${getShipCount(player)}`, x, y + 2 * scoreboardLineYDiffPx);
    return y + 3 * scoreboardLineYDiffPx;
  };

  const scoreboardShipSizePx = scoreboardFontSizePx * 1.7;
  const drawShip = (ctx: any, playerIndex: number, x: number, y: number, iconSize = scoreboardShipSizePx) =>
    ctx.drawImage(bufferCanvas, 500 + 100 * playerIndex, 0, fixedCellSize, fixedCellSize, x, y, iconSize, iconSize);
  const drawShipYard = (ctx: any, playerIndex: number, x: number, y: number, iconSize = scoreboardShipSizePx) =>
    ctx.drawImage(bufferCanvas, 500 + 100 * playerIndex, 400, fixedCellSize, fixedCellSize, x, y, iconSize, iconSize);

  const scoreboardShipXPaddingPx = scoreboardShipSizePx + scoreboardPaddingPx;
  const drawShipAndYardCounts = (
    ctx: any,
    player: any,
    playerIndex: number,
    x: number,
    y: number,
    iconSize = scoreboardShipSizePx
  ) => {
    drawShip(ctx, playerIndex, x, y);
    ctx.fillText(`x ${getNumFleets(player)}`, x + scoreboardShipXPaddingPx, y + 0.28 * iconSize);
    drawShipYard(ctx, playerIndex, x, y + iconSize);
    ctx.fillText(`x ${getNumShipyards(player)}`, x + scoreboardShipXPaddingPx, y + 1.38 * iconSize);
    return y + 2.38 * iconSize;
  };

  const drawFleetLaunches = (ctx: any, playerIndex: number, x: number, y: number, reverse?: boolean) => {
    const actions = steps[step][playerIndex].action || {};
    const launches = (Object.values(actions) as string[])
      .filter((a: string) => a.includes('LAUNCH'))
      .map((a: string) => a.substring(7).replace(/_/, ' '))
      .sort((a, b) => (parseInt(a.split(' ')[0]) < parseInt(b.split(' ')[0]) ? 1 : -1));
    if (launches.length > 0) {
      ctx.fillText('Launches:', x, y);
    }
    const dir = reverse ? -1 : 1;
    for (let i = 0; i < Math.min(launches.length, 5); i++) {
      ctx.fillText(launches[i], x, y + dir * (i + 1) * scoreboardLineYDiffPx);
    }
    if (launches.length > 5) {
      ctx.fillText(`+ ${launches.length - 5} more...`, x, y + dir * 6 * scoreboardLineYDiffPx);
    }
  };

  // Render Scoreboard for each player, if we have enough room on the sides of the window.
  if (width / height >= 1.3) {
    fgCtx.fillStyle = '#FFFFFF';
    fgCtx.font = `normal ${scoreboardFontSizePx}px sans-serif`;
    fgCtx.textBaseline = 'top';
    fgCtx.textAlign = 'left';
    const topStartY = topLeftCell.dy;
    const bottomStartY = botRightCell.dy + botRightCell.ds - 2 * scoreboardShipSizePx - 2 * scoreboardLineYDiffPx;
    players.forEach((player: any, playerIndex: number) => {
      const x =
        playerIndex % 2 === 1
          ? Math.max(
              // Make sure we don't start within the game area on the right side.
              botRightCell.dx + botRightCell.ds + 2 * boxPadding,
              width - topLeftCell.dy - 5.5 * scoreboardFontSizePx
            )
          : topLeftCell.dy;
      const startY = playerIndex < 2 ? topStartY : bottomStartY;
      const nextY = writeScoreboardText(fgCtx, player, x, startY);
      const actionY = drawShipAndYardCounts(fgCtx, player, playerIndex, x, nextY);
      if (playerIndex > 1) {
        drawFleetLaunches(fgCtx, playerIndex, x, startY - scoreboardLineYDiffPx, true);
      } else {
        drawFleetLaunches(fgCtx, playerIndex, x, actionY);
      }
    });
  }

  // Populate the legend which renders agent icons and names (see player.html).
  if (agents && agents.length && (!agents[0].color || !agents[0].image)) {
    const getPieceImage = (playerIndex: number) => {
      const pieceCanvas: any = document.createElement('canvas');
      parent.appendChild(pieceCanvas);
      pieceCanvas.style.marginLeft = '10000px';
      pieceCanvas.width = 100;
      pieceCanvas.height = 100;
      const ctx = pieceCanvas.getContext('2d');
      drawShip(ctx, playerIndex, 0, 0, 100);
      const dataUrl = pieceCanvas.toDataURL();
      parent.removeChild(pieceCanvas);
      return dataUrl;
    };

    agents.forEach((agent: any) => {
      agent.color = '#FFFFFF';
      agent.image = getPieceImage(agent.index);
    });
    // update({ agents });
  }
}
