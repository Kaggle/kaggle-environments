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
      // Common Dimensions.
      // Game dimensions: 96x72
      // We also add 'borders'
  const unit = Math.floor(Math.min(height / (72+2), width / (96+2)));
  const offsetX = Math.floor((width % 98)/2);
  const offsetY = Math.floor((height % 74)/2);

  // the field's top-left corner starts at (offsetX, offsetY)
  const leftX = offsetX;
  const topY = offsetY;
  const rightX = offsetX + 98*unit;
  const bottomY = offsetY + 74*unit;

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

  const drawCell = ({x, y, color}) => {
    const posX = x*unit + leftX;
    const posY = y*unit + topY;
    drawLine({x1: posX, y1:posY, x2:posX+unit, y2:posY, style: {lineWidth:unit, strokeStyle: color}});
  };

  // drawBorders
  drawLine({x1:leftX, y1:topY, x2:rightX, y2:topY, style:{lineWidth:unit, strokeStyle:"#ffffff"}});
  drawLine({x1:leftX, y1:bottomY, x2:rightX, y2:bottomY, style:{lineWidth:unit, strokeStyle:"#ffffff"}});

  drawLine({x1:leftX, y1:topY, x2:leftX, y2:bottomY, style:{lineWidth:unit, strokeStyle:"#ffffff"}});
  drawLine({x1:rightX, y1:topY, x2:rightX, y2:bottomY, style:{lineWidth:unit, strokeStyle:"#ffffff"}});

  // drawCell({x:10, y:10, color:"#0000ff"});

  const minimap = environment.steps[step][0].observation.minimap;
  var x,y,z;
  var i = 0;
  var colors = ["black", "red", "yellow", "purple"];

  for (y = 0; y<72; y++) {
    for (x = 0; x<96; x++) {
      for (z = 0; z < 4; z++) {
        if (minimap[i] > 0) {
          drawCell({x:x, y:y, color:colors[z]});
        }
        i += 1;
      }
    }

  }
  }
