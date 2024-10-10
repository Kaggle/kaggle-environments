async function renderer(context) {
  const {
    environment,
    frame,
    height = 400,
    parent,
    step,
    width = 400,
  } = context;

  // Common Dimensions.
  const canvasSize = Math.min(height, width);
  const boardSize = canvasSize * 0.8;
  const squareSize = boardSize / 8;
  const offset = (canvasSize - boardSize) / 2;

  // Canvas Setup.
  let canvas = parent.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    parent.appendChild(canvas);
  }

  // Canvas setup and reset.
  let c = canvas.getContext("2d");
  canvas.width = canvasSize;
  canvas.height = canvasSize;
  c.clearRect(0, 0, canvas.width, canvas.height);
  // Draw the Chessboard
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const x = col * squareSize + offset;
      const y = row * squareSize + offset;

      // Alternate colors for squares
      c.fillStyle = (row + col) % 2 === 0 ? "#D18B47" : "#FFCE9E";
      c.fillRect(x, y, squareSize, squareSize);
    }
  }

  // Draw the Pieces
  const board = environment.steps[step][0].observation.board;
  const chess = new Chess(board);
  const boardObj = chess.board();

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const piece = boardObj[row][col];
      if (piece) {
        const x = col * squareSize + offset;
        const y = row * squareSize + offset;
        drawPiece(c, piece.type, piece.color, x, y, squareSize);
      }
    }
  }
}

// Helper function to draw individual pieces (replace with your own images/logic)
function drawPiece(c, type, color, x, y, size) {
  const pieceCode = color === "w" ? type.toUpperCase() : type.toLowerCase();
  // Unicode characters for chess pieces
  const pieceSymbols = {
    P: "♙",
    R: "♖",
    N: "♘",
    B: "♗",
    Q: "♕",
    K: "♔",
    p: "♟",
    r: "♜",
    n: "♞",
    b: "♝",
    q: "♛",
    k: "♚",
  };

  c.font = `${size * 0.8}px Arial`;
  c.textAlign = "center";
  c.textBaseline = "middle";
  c.fillStyle = color === "w" ? "white" : "black";
  c.fillText(pieceSymbols[pieceCode], x + size / 2, y + size / 2);
}
