import { RendererOptions, ReplayData } from '@kaggle-environments/core';
import { Chess } from 'chess.js';
import { MOVES, OPENINGS, pieceImagesSrc } from './consts';

// --- Type Definitions ---

interface ChessObservation {
  board: string;
  mark: 'white' | 'black';
  lastMove?: string;
  remainingOverageTime?: number;
  opponentRemainingOverageTime?: number;
  step?: number;
}

interface ChessAgentStep {
  action: string;
  reward: number | null;
  observation: ChessObservation;
  status: string;
  info: Record<string, any>;
}

type ChessStep = [ChessAgentStep, ChessAgentStep];

interface ChessReplayData extends ReplayData<ChessStep[]> {
  rewards?: (number | null | undefined)[];
  info?: {
    TeamNames?: string[];
    [key: string]: any;
  };
  viewer?: any;
}

// --- Module Scoped Helpers (Moved outside renderer) ---

const pieceImages: Record<string, HTMLImageElement> = {};

function initializePieceImages() {
  const pieces = ['P', 'R', 'N', 'B', 'Q', 'K'];
  pieces.forEach((piece) => {
    const whiteChar = piece.toLowerCase();
    const blackChar = piece.toUpperCase();
    const whiteImg = new Image();
    const blackImg = new Image();

    // Check if pieceImagesSrc is defined to avoid runtime errors if consts file is incomplete
    if (pieceImagesSrc) {
      whiteImg.src = pieceImagesSrc[whiteChar];
      blackImg.src = pieceImagesSrc[blackChar];
      pieceImages[whiteChar] = whiteImg;
      pieceImages[blackChar] = blackImg;
    }
  });
}

// Initialize immediately when module loads
initializePieceImages();

function drawPiece(c: CanvasRenderingContext2D, type: string, color: string, x: number, y: number, size: number) {
  const pieceCode = color === 'w' ? type.toLowerCase() : type.toUpperCase();
  const img = pieceImages[pieceCode];

  if (img) {
    c.drawImage(img, x, y, size, size);
  } else {
    const pieceSymbols: Record<string, string> = {
      P: '♙',
      R: '♖',
      N: '♘',
      B: '♗',
      Q: '♕',
      K: '♔',
      p: '♟',
      r: '♜',
      n: '♞',
      b: '♝',
      q: '♛',
      k: '♚',
    };

    c.font = `${size * 0.8}px Arial`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillStyle = color === 'w' ? 'white' : 'black';
    c.fillText(pieceSymbols[pieceCode], x + size / 2, y + size / 2);
  }
}

// --- Main Renderer Function ---

export function renderer(context: RendererOptions<ChessStep[]>) {
  const environment = context.replay as ChessReplayData;
  const { parent, step } = context;
  const width = parent.clientWidth || 400;
  const height = parent.clientHeight || 400;

  // Common Dimensions.
  const canvasSize = Math.min(height, width);
  const boardSize = canvasSize * 0.8;
  const squareSize = boardSize / 8;
  const offset = (canvasSize - boardSize) / 2;

  // Canvas Setup.
  let canvas = parent.querySelector('canvas') as HTMLCanvasElement | null;
  if (!canvas) {
    canvas = document.createElement('canvas');
    parent.appendChild(canvas);
  }

  // Create the Download PGN button
  let downloadButton = parent.querySelector('#copy-pgn') as HTMLElement | null;

  if (!downloadButton && environment.steps.length) {
    try {
      // Guard: Ensure step 0 exists
      const firstStepAgent = environment.steps[0]?.[0];
      if (!firstStepAgent) throw new Error('No initial state found');

      const board = firstStepAgent.observation.board;
      const info = environment.info;
      const agent1 = info?.TeamNames?.[0] || 'Agent 1';
      const agent2 = info?.TeamNames?.[1] || 'Agent 2';
      const game = new Chess();

      let result = environment.rewards ?? [];
      if (result.some((r) => r === undefined || r === null)) {
        result = result.map((r) => (r === undefined || r === null ? 0 : 1));
      }

      (game as any).header(
        'Event',
        'FIDE & Google Efficient Chess AI Challenge (https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge)',
        'White',
        agent1,
        'Black',
        agent2
      );

      const openingIdx = OPENINGS.indexOf(board);
      if (openingIdx !== -1 && MOVES[openingIdx]) {
        const moves = MOVES[openingIdx].split(' ');
        for (let i = 0; i < moves.length; i++) {
          const move = moves[i];
          game.move({ from: move.slice(0, 2), to: move.slice(2, 4) });
        }
      }

      for (let i = 1; i < environment.steps.length; i++) {
        const move = environment.steps[i][(i - 1) % 2].action;

        if (!move) continue;

        if (move.length === 4) {
          game.move({ from: move.slice(0, 2), to: move.slice(2, 4) });
        } else if (move.length === 5) {
          game.move({
            from: move.slice(0, 2),
            to: move.slice(2, 4),
            promotion: move.slice(4, 5),
          });
        }
      }

      let pgn = game.pgn();

      if (pgn.indexOf(' 0-0') !== -1) {
        pgn = pgn.split(' 0-0')[0];
      }

      downloadButton = document.createElement('button');
      downloadButton.id = 'copy-pgn';
      downloadButton.textContent = 'Copy PGN';
      downloadButton.style.position = 'absolute';
      downloadButton.style.top = '10px';
      downloadButton.style.left = '10px';
      downloadButton.style.zIndex = '1';

      if (!environment.viewer) {
        parent.appendChild(downloadButton);
      }

      downloadButton.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(pgn);
          alert('PGN Copied');
          return;
        } catch {
          console.info('Clipboard access failed. Fall back to display for manual copy.');
        }

        try {
          const btn = document.getElementById('copy-pgn') as HTMLElement;
          if (btn) btn.textContent = '';

          const pgnDiv = document.createElement('div');
          pgnDiv.style.position = 'absolute';
          pgnDiv.style.top = '8px';
          pgnDiv.style.left = '8px';
          pgnDiv.style.zIndex = '2';
          pgnDiv.style.border = '1px solid black';
          pgnDiv.style.padding = '8px';
          pgnDiv.style.background = '#FFFFFF';
          pgnDiv.style.fontFamily = 'monospace';
          pgnDiv.style.whiteSpace = 'pre-wrap';

          const pgnLines = pgn.split('\n');
          pgnLines.forEach((line) => {
            const lineSpan = document.createElement('span');
            lineSpan.textContent = line + '\n';
            pgnDiv.appendChild(lineSpan);
          });
          parent.appendChild(pgnDiv);

          const closeButton = document.createElement('span');
          closeButton.textContent = '×';
          closeButton.style.position = 'absolute';
          closeButton.style.top = '5px';
          closeButton.style.right = '5px';
          closeButton.style.cursor = 'pointer';
          closeButton.style.float = 'right';
          closeButton.style.fontSize = '16px';
          closeButton.style.marginLeft = '5px';

          closeButton.addEventListener('click', () => {
            if (btn) btn.textContent = 'Copy PGN';
            parent.removeChild(pgnDiv);
          });
          pgnDiv.appendChild(closeButton);
        } catch {
          console.error('Cannot display div');
          alert('PGN cannot be generated');
        }
      });
    } catch (e) {
      console.error('Cannot create game pgn');
      console.error(e);
    }
  }

  // Canvas setup and reset.
  const c = canvas.getContext('2d');
  if (!c) return;

  canvas.width = canvasSize;
  canvas.height = canvasSize;
  c.clearRect(0, 0, canvas.width, canvas.height);

  // Draw the Chessboard
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const x = col * squareSize + offset;
      const y = row * squareSize + offset;

      c.fillStyle = (row + col) % 2 === 0 ? '#FFCE9E' : '#D18B47';
      c.fillRect(x, y, squareSize, squareSize);
    }
  }

  // Guard: Ensure current step data exists
  const currentStep = environment.steps[step];
  const currentAgent0 = currentStep?.[0];

  if (!currentStep || !currentAgent0) return;

  // Draw the team names and game status
  if (!environment.viewer) {
    const info = environment.info;
    const agent1 = info?.TeamNames?.[0] || 'Agent 1';
    const agent2 = info?.TeamNames?.[1] || 'Agent 2';

    const firstGame = currentAgent0.observation.mark == 'white';

    const fontSize = Math.round(0.33 * offset);
    c.font = `${fontSize}px sans-serif`;
    c.fillStyle = '#FFFFFF';

    const agent1Reward = currentStep[0]?.reward ?? 0;
    const agent2Reward = currentStep[1]?.reward ?? 0;

    const charCount = agent1.length + agent2.length + 12;
    const title = `${firstGame ? '\u25A0' : '\u25A1'}${agent1} (${agent1Reward}) vs ${
      firstGame ? '\u25A1' : '\u25A0'
    }${agent2} (${agent2Reward})`;
    c.fillText(title, offset + 4 * squareSize - Math.floor((charCount * fontSize) / 4), 40);
  }

  // Draw the Pieces
  const board = currentAgent0.observation.board;
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
