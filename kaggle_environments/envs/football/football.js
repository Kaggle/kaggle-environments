function renderer(context) {
    const {
    act,
    agents,
    environment,
    frame,
    height = 800,
    interactive,
    isInteractive,
    parent,
    step,
    update,
    width = 800,
    } = context;
  CELL_Y = 72;
  CELL_X = 96;
  const unit = Math.floor(Math.min(height / (CELL_Y+2), width / (CELL_X+2)));
  const offsetX = Math.floor((width % (CELL_X + 2))/2);
  const offsetY = Math.floor((height % (CELL_Y + 2))/2);

  // the field's top-left corner starts at (offsetX, offsetY)
  const leftX = offsetX;
  const topY = offsetY;
  const rightX = offsetX + (CELL_X + 2)*unit;
  const bottomY = offsetY + (CELL_Y + 2)*unit;

  // Canvas Setup.
  let canvas = parent.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    parent.appendChild(canvas);
  }
  canvas.style.cursor = "default";

  // Canvas setup and reset.
  let c = canvas.getContext("2d");
  canvas.width = width;
  canvas.height = height;
  c.clearRect(0, 0, canvas.width, canvas.height);
  c.fillStyle = "#7cfc00"
  c.fillRect(0, 0, canvas.width, canvas.height);

  const drawStyle = ({
    lineWidth = 1,
    lineCap,
    strokeStyle = "#FFF",
    shadow,
  }) => {
    c.lineWidth = lineWidth;
    c.strokeStyle = strokeStyle;
    if (lineCap) c.lineCap = lineCap;
    if (shadow) {
      c.shadowOffsetX = shadow.offsetX || 0;
      c.shadowOffsetY = shadow.offsetY || 0;
      c.shadowColor = shadow.color || strokeStyle;
      c.shadowBlur = shadow.blur || 0;
    }
  };

  const drawLine = ({ x1, y1, x2, y2, style }) => {
    c.beginPath();
    drawStyle(style || {});
    c.moveTo((x1 || 0), (y1 || 0));
    c.lineTo((x2 || x1), (y2 || y1));
    c.stroke();
  };

  const drawCell = ({ x, y, color }) => {
    const centerX = (((x + 1) / 2) * CELL_X)*unit + leftX + unit;
    const centerY = (((y + 1) / 2) * CELL_Y)*unit + topY + unit;

    const posX = centerX - unit / 2;
    const posY = centerY;
    drawLine({x1: posX, y1:posY, x2:posX+unit, y2:posY, style: {lineWidth:unit, strokeStyle: color}});
  };

  // drawBorders
  drawLine({x1:leftX, y1:topY, x2:rightX, y2:topY, style:{lineWidth:unit, strokeStyle:"#ffffff"}});
  drawLine({x1:leftX, y1:bottomY, x2:rightX, y2:bottomY, style:{lineWidth:unit, strokeStyle:"#ffffff"}});

  drawLine({x1:leftX, y1:topY, x2:leftX, y2:bottomY, style:{lineWidth:unit, strokeStyle:"#ffffff"}});
  drawLine({x1:rightX, y1:topY, x2:rightX, y2:bottomY, style:{lineWidth:unit, strokeStyle:"#ffffff"}});


  const raw_view_player0 = environment.steps[step][0].observation.players_raw[0];
  for (let i = 0; i < raw_view_player0['left_team'].length; i++) {
    const entry = raw_view_player0['left_team'][i];
    drawCell({ x: entry[0], y: entry[1], color:"black" });
  }
  for (let i = 0; i < raw_view_player0['right_team'].length; i++) {
    const entry = raw_view_player0['right_team'][i];
    drawCell({ x: entry[0], y: entry[1], color:"red" });
  }
  drawCell({ x: raw_view_player0['ball'][0], y: raw_view_player0['ball'][1], color:"yellow" });
}
