async function renderer(context) {
  const { environment, step, parent, width = 800, height = 600 } = context;

  // Wall bitfield constants
  var WALL_N = 1;
  var WALL_E = 2;
  var WALL_S = 4;
  var WALL_W = 8;
  var PETRIFIED = 16;

  // Robot type constants
  var FACTORY = 0;
  var SCOUT = 1;
  var WORKER = 2;
  var MINER = 3;

  var PLAYER_COLORS = ['#42A5F5', '#EF5350'];
  var PLAYER_COLORS_LIGHT = ['rgba(66,165,245,0.25)', 'rgba(239,83,80,0.25)'];
  var TYPE_LABELS = { 0: 'F', 1: 'S', 2: 'W', 3: 'M' };

  var currentStep = environment.steps[step];
  if (!currentStep || !currentStep[0]) return;

  var obs = currentStep[0].observation;
  var config = environment.configuration;
  var gridWidth = config.width || 20;
  var southBound = obs.southBound || 0;
  var northBound = obs.northBound || 29;
  var windowHeight = northBound - southBound + 1;

  // Parse global state
  var globalWalls = obs.globalWalls || {};
  var globalCrystals = obs.globalCrystals || {};
  var globalRobots = obs.globalRobots || {};
  var globalMines = obs.globalMines || {};
  var globalMiningNodes = obs.globalMiningNodes || {};

  var VISION = {
    0: config.visionFactory || 4,
    1: config.visionScout || 5,
    2: config.visionWorker || 3,
    3: config.visionMiner || 3,
  };

  var robots = [];
  for (var uid in globalRobots) {
    var d = globalRobots[uid];
    robots.push({ type: d[0], col: d[1], row: d[2], energy: d[3], owner: d[4] });
  }

  // Compute per-player visible cells
  var visible = [{}, {}]; // visible[owner]["col,row"] = true
  for (var vi = 0; vi < robots.length; vi++) {
    var vr = robots[vi];
    var vRange = VISION[vr.type] || 3;
    for (var vdc = -vRange; vdc <= vRange; vdc++) {
      for (var vdr = -vRange; vdr <= vRange; vdr++) {
        if (Math.abs(vdc) + Math.abs(vdr) <= vRange) {
          var vc = vr.col + vdc;
          if (vc >= 0 && vc < gridWidth) {
            visible[vr.owner][vc + ',' + (vr.row + vdr)] = true;
          }
        }
      }
    }
  }

  var mines = {};
  for (var key in globalMines) {
    var m = globalMines[key];
    mines[key] = { energy: m[0], maxEnergy: m[1], owner: m[2] };
  }

  // Compute energy totals
  var totalEnergy = [0, 0];
  var robotCounts = [
    { F: 0, S: 0, W: 0, M: 0 },
    { F: 0, S: 0, W: 0, M: 0 },
  ];
  for (var i = 0; i < robots.length; i++) {
    var r = robots[i];
    totalEnergy[r.owner] += r.energy;
    robotCounts[r.owner][TYPE_LABELS[r.type]]++;
  }

  // Setup canvas
  var canvas = parent.querySelector('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    parent.appendChild(canvas);
  }
  canvas.width = width;
  canvas.height = height;

  var c = canvas.getContext('2d');
  if (!c) return;

  // Reserve space for header and status text
  var headerH = 30;
  var statusH = 20;
  var canvasH = height - headerH - statusH;
  var canvasW = width;

  // Calculate cell size
  var cellW = canvasW / gridWidth;
  var cellH = canvasH / windowHeight;
  var cellSize = Math.min(cellW, cellH);
  var gridW = cellSize * gridWidth;
  var gridH = cellSize * windowHeight;
  var offsetX = (canvasW - gridW) / 2;
  var offsetY = headerH + (canvasH - gridH) / 2;

  function cellToCanvas(col, row) {
    var gridRow = northBound - row;
    return { x: offsetX + col * cellSize, y: offsetY + gridRow * cellSize };
  }

  // Clear — dark background
  c.fillStyle = '#1a1a2e';
  c.fillRect(0, 0, canvas.width, canvas.height);

  // Draw header
  c.font = 'bold 12px sans-serif';
  c.textBaseline = 'top';
  var p0 =
    'P1: E=' +
    totalEnergy[0] +
    ' F:' +
    robotCounts[0].F +
    ' S:' +
    robotCounts[0].S +
    ' W:' +
    robotCounts[0].W +
    ' M:' +
    robotCounts[0].M;
  var p1 =
    'P2: E=' +
    totalEnergy[1] +
    ' F:' +
    robotCounts[1].F +
    ' S:' +
    robotCounts[1].S +
    ' W:' +
    robotCounts[1].W +
    ' M:' +
    robotCounts[1].M;
  c.fillStyle = PLAYER_COLORS[0];
  c.textAlign = 'left';
  c.fillText(p0, 8, 8);
  c.fillStyle = PLAYER_COLORS[1];
  c.textAlign = 'right';
  c.fillText(p1, canvas.width - 8, 8);

  // Draw cells
  for (var row = southBound; row <= northBound; row++) {
    var rowWalls = globalWalls[String(row)];
    for (var col = 0; col < gridWidth; col++) {
      var pos = cellToCanvas(col, row);
      var x = pos.x,
        y = pos.y;
      var w = rowWalls ? rowWalls[col] : 0;

      if (w === PETRIFIED) {
        c.fillStyle = '#3a2a1a';
        c.fillRect(x, y, cellSize, cellSize);
        c.strokeStyle = '#5a3a1a';
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(x, y);
        c.lineTo(x + cellSize, y + cellSize);
        c.moveTo(x + cellSize, y);
        c.lineTo(x, y + cellSize);
        c.stroke();
        continue;
      }

      // Cell background
      c.fillStyle = 'rgba(255, 255, 255, 0.03)';
      c.fillRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);

      // Walls
      c.strokeStyle = '#556677';
      c.lineWidth = 2;
      if (w & WALL_N) {
        c.beginPath();
        c.moveTo(x, y);
        c.lineTo(x + cellSize, y);
        c.stroke();
      }
      if (w & WALL_S) {
        c.beginPath();
        c.moveTo(x, y + cellSize);
        c.lineTo(x + cellSize, y + cellSize);
        c.stroke();
      }
      if (w & WALL_E) {
        c.beginPath();
        c.moveTo(x + cellSize, y);
        c.lineTo(x + cellSize, y + cellSize);
        c.stroke();
      }
      if (w & WALL_W) {
        c.beginPath();
        c.moveTo(x, y);
        c.lineTo(x, y + cellSize);
        c.stroke();
      }
    }
  }

  // Center divider
  var divX = offsetX + (gridWidth / 2) * cellSize;
  c.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  c.lineWidth = 1;
  c.setLineDash([4, 4]);
  c.beginPath();
  c.moveTo(divX, offsetY);
  c.lineTo(divX, offsetY + gridH);
  c.stroke();
  c.setLineDash([]);

  // Draw mining nodes
  for (var nkey in globalMiningNodes) {
    var nparts = nkey.split(',');
    var ncol = parseInt(nparts[0]),
      nrow = parseInt(nparts[1]);
    if (nrow < southBound || nrow > northBound) continue;
    var np = cellToCanvas(ncol, nrow);
    var nx = np.x,
      ny = np.y;
    // Subtle background highlight
    c.fillStyle = 'rgba(205, 133, 63, 0.15)';
    c.fillRect(nx + 1, ny + 1, cellSize - 2, cellSize - 2);
    // Diamond outline
    var ncx = nx + cellSize / 2,
      ncy = ny + cellSize / 2;
    var nr = cellSize * 0.2;
    c.strokeStyle = '#CD853F';
    c.lineWidth = Math.max(1.5, cellSize * 0.06);
    c.beginPath();
    c.moveTo(ncx, ncy - nr);
    c.lineTo(ncx + nr, ncy);
    c.lineTo(ncx, ncy + nr);
    c.lineTo(ncx - nr, ncy);
    c.closePath();
    c.stroke();
    // Small dot in center
    c.fillStyle = '#CD853F';
    c.beginPath();
    c.arc(ncx, ncy, cellSize * 0.06, 0, Math.PI * 2);
    c.fill();
  }

  // Draw crystals
  for (var ckey in globalCrystals) {
    var parts = ckey.split(',');
    var ccol = parseInt(parts[0]),
      crow = parseInt(parts[1]);
    if (crow < southBound || crow > northBound) continue;
    var cp = cellToCanvas(ccol, crow);
    var cx = cp.x + cellSize / 2,
      cy = cp.y + cellSize / 2;
    var cr = cellSize * 0.18;
    c.fillStyle = 'rgba(255, 215, 0, 0.7)';
    c.strokeStyle = '#FFD700';
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(cx, cy - cr * 1.2);
    c.lineTo(cx + cr, cy);
    c.lineTo(cx, cy + cr * 0.8);
    c.lineTo(cx - cr, cy);
    c.closePath();
    c.fill();
    c.stroke();
    if (cellSize > 16) {
      c.fillStyle = '#FFD700';
      c.font = Math.max(7, cellSize * 0.22) + 'px sans-serif';
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(String(globalCrystals[ckey]), cx, cy + cr * 1.5);
    }
  }

  // Draw mines
  for (var mkey in mines) {
    var mp = mkey.split(',');
    var mcol = parseInt(mp[0]),
      mrow = parseInt(mp[1]);
    if (mrow < southBound || mrow > northBound) continue;
    var mpos = cellToCanvas(mcol, mrow);
    var mine = mines[mkey];
    var mx = mpos.x + cellSize / 2,
      my = mpos.y + cellSize / 2;
    var mr = cellSize * 0.25;
    var mcolor = PLAYER_COLORS[mine.owner];
    c.fillStyle = mcolor;
    c.globalAlpha = 0.4;
    c.beginPath();
    c.moveTo(mx, my - mr);
    c.lineTo(mx + mr, my + mr * 0.7);
    c.lineTo(mx - mr, my + mr * 0.7);
    c.closePath();
    c.fill();
    c.globalAlpha = 1;
    c.strokeStyle = mcolor;
    c.lineWidth = 1.5;
    c.stroke();
    var mpct = mine.maxEnergy > 0 ? mine.energy / mine.maxEnergy : 0;
    var mbarW = cellSize * 0.5,
      mbarH = Math.max(2, cellSize * 0.06);
    var mbarX = mx - mbarW / 2,
      mbarY = mpos.y + cellSize - mbarH - 1;
    c.fillStyle = '#333';
    c.fillRect(mbarX, mbarY, mbarW, mbarH);
    c.fillStyle = '#FFD700';
    c.fillRect(mbarX, mbarY, mbarW * mpct, mbarH);
  }

  // Draw robots (factories first)
  var sortedRobots = robots.slice().sort(function (a, b) {
    if (a.type === FACTORY && b.type !== FACTORY) return -1;
    if (a.type !== FACTORY && b.type === FACTORY) return 1;
    return 0;
  });

  for (var ri = 0; ri < sortedRobots.length; ri++) {
    var robot = sortedRobots[ri];
    if (robot.row < southBound || robot.row > northBound) continue;
    var rp = cellToCanvas(robot.col, robot.row);
    var rx = rp.x,
      ry = rp.y;
    var color = PLAYER_COLORS[robot.owner];
    var lightColor = PLAYER_COLORS_LIGHT[robot.owner];
    var rcx = rx + cellSize / 2,
      rcy = ry + cellSize / 2;
    var rr = cellSize * 0.35;

    if (robot.type === FACTORY) {
      var s = cellSize * 0.6;
      c.fillStyle = lightColor;
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.fillRect(rcx - s / 2, rcy - s / 2, s, s);
      c.strokeRect(rcx - s / 2, rcy - s / 2, s, s);
      var notch = s * 0.15;
      c.fillStyle = color;
      c.fillRect(rcx - notch, rcy - s / 2 - notch, notch * 2, notch);
      c.fillRect(rcx - notch, rcy + s / 2, notch * 2, notch);
      c.fillRect(rcx - s / 2 - notch, rcy - notch, notch, notch * 2);
      c.fillRect(rcx + s / 2, rcy - notch, notch, notch * 2);
    } else if (robot.type === SCOUT) {
      c.fillStyle = lightColor;
      c.strokeStyle = color;
      c.lineWidth = 1.5;
      c.beginPath();
      c.moveTo(rcx, rcy - rr);
      c.lineTo(rcx + rr, rcy);
      c.lineTo(rcx, rcy + rr);
      c.lineTo(rcx - rr, rcy);
      c.closePath();
      c.fill();
      c.stroke();
    } else if (robot.type === WORKER) {
      c.fillStyle = lightColor;
      c.strokeStyle = color;
      c.lineWidth = 1.5;
      c.beginPath();
      for (var hi = 0; hi < 6; hi++) {
        var angle = (Math.PI / 3) * hi - Math.PI / 6;
        var px = rcx + rr * Math.cos(angle);
        var py = rcy + rr * Math.sin(angle);
        if (hi === 0) c.moveTo(px, py);
        else c.lineTo(px, py);
      }
      c.closePath();
      c.fill();
      c.stroke();
    } else if (robot.type === MINER) {
      c.fillStyle = lightColor;
      c.strokeStyle = color;
      c.lineWidth = 1.5;
      c.beginPath();
      c.arc(rcx, rcy, rr, 0, Math.PI * 2);
      c.closePath();
      c.fill();
      c.stroke();
    }

    // Type letter
    c.fillStyle = color;
    c.font = 'bold ' + Math.max(10, cellSize * 0.35) + 'px sans-serif';
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(TYPE_LABELS[robot.type], rcx, rcy);

    // Energy bar
    var barW = cellSize * 0.7,
      barH = Math.max(2, cellSize * 0.08);
    var barX = rcx - barW / 2,
      barY = ry + cellSize - barH - 1;
    var maxE = robot.type === FACTORY ? 1000 : robot.type === SCOUT ? 50 : robot.type === WORKER ? 200 : 300;
    var pct = Math.min(1, robot.energy / maxE);
    c.fillStyle = '#333';
    c.fillRect(barX, barY, barW, barH);
    c.fillStyle = pct > 0.3 ? '#4CAF50' : pct > 0.1 ? '#FF9800' : '#F44336';
    c.fillRect(barX, barY, barW * pct, barH);
  }

  // Fog of war overlay — darken cells not visible to either player
  for (var frow = southBound; frow <= northBound; frow++) {
    for (var fcol = 0; fcol < gridWidth; fcol++) {
      var fkey = fcol + ',' + frow;
      var p0vis = visible[0][fkey];
      var p1vis = visible[1][fkey];
      if (!p0vis && !p1vis) {
        var fp = cellToCanvas(fcol, frow);
        c.fillStyle = 'rgba(0, 0, 0, 0.6)';
        c.fillRect(fp.x, fp.y, cellSize, cellSize);
      } else if (p0vis && !p1vis) {
        // Only P1 sees — tint blue
        var fp = cellToCanvas(fcol, frow);
        c.fillStyle = 'rgba(33, 150, 243, 0.08)';
        c.fillRect(fp.x, fp.y, cellSize, cellSize);
      } else if (!p0vis && p1vis) {
        // Only P2 sees — tint red
        var fp = cellToCanvas(fcol, frow);
        c.fillStyle = 'rgba(244, 67, 54, 0.08)';
        c.fillRect(fp.x, fp.y, cellSize, cellSize);
      }
    }
  }

  // Scroll boundary indicator at bottom
  c.fillStyle = 'rgba(244, 67, 54, 0.12)';
  var boundaryY = offsetY + gridH - cellSize;
  c.fillRect(offsetX, boundaryY, gridW, cellSize);
  c.strokeStyle = 'rgba(244, 67, 54, 0.5)';
  c.lineWidth = 2;
  c.setLineDash([6, 3]);
  c.beginPath();
  c.moveTo(offsetX, boundaryY);
  c.lineTo(offsetX + gridW, boundaryY);
  c.stroke();
  c.setLineDash([]);

  // Row labels
  c.fillStyle = '#888';
  c.font = Math.max(8, cellSize * 0.25) + 'px sans-serif';
  c.textAlign = 'right';
  c.textBaseline = 'middle';
  var labelStep = Math.max(1, Math.floor(windowHeight / 10));
  for (var lr = southBound; lr <= northBound; lr += labelStep) {
    var lp = cellToCanvas(0, lr);
    c.fillText(String(lr), offsetX - 4, lp.y + cellSize / 2);
  }

  // Status bar
  var isGameOver = currentStep.every(function (s) {
    return s.status === 'DONE';
  });
  c.font = '11px sans-serif';
  c.textBaseline = 'bottom';
  c.fillStyle = '#aaa';
  c.textAlign = 'center';
  var statusText = 'Step ' + (obs.step || step) + '  |  Scroll: ' + southBound + '-' + northBound;
  if (isGameOver) {
    var r0 = currentStep[0].reward,
      r1 = currentStep[1].reward;
    var result = 'Draw';
    if (r0 !== null && r1 !== null) {
      if (r0 > r1) result = 'P1 wins!';
      else if (r1 > r0) result = 'P2 wins!';
    }
    statusText = result + '  |  ' + statusText;
  } else {
    var mineCount = Object.keys(mines).length;
    var crystalCount = Object.keys(globalCrystals).length;
    statusText += '  |  Mines: ' + mineCount + '  |  Crystals: ' + crystalCount;
  }
  c.fillText(statusText, canvas.width / 2, canvas.height - 4);
}
