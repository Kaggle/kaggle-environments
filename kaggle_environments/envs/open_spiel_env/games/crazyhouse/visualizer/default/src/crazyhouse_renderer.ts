import { RendererOptions } from '@kaggle-environments/core';
import {
  DARK_SQUARE_COLOR,
  DEFAULT_NUM_COLS,
  DEFAULT_NUM_ROWS,
  LIGHT_SQUARE_COLOR,
  PIECE_IMAGES_SRC,
} from '../../../../chess/visualizer/default/src/consts';

interface CrazyhouseObservation {
  fen: string;
  board: string[][];
  side_to_move: 'w' | 'b';
  castling_rights: string;
  en_passant: string;
  halfmove_clock: number;
  fullmove_number: number;
  pockets: { white: Record<string, number>; black: Record<string, number> };
  current_player: 'white' | 'black' | number;
  is_terminal: boolean;
  winner: 'white' | 'black' | 'draw' | null;
}

const POCKET_PIECE_ORDER = ['Q', 'R', 'B', 'N', 'P'];
const POCKET_PIECE_SIZE = 28;
const POCKET_BG = '#f0d9b5';
const POCKET_BORDER = '#b58863';

function parseObservation(step: any): CrazyhouseObservation | null {
  const raw = step?.[0]?.observation?.observationString;
  if (!raw || typeof raw !== 'string') return null;
  try {
    return JSON.parse(raw) as CrazyhouseObservation;
  } catch {
    return null;
  }
}

function buildPocketHtml(pocket: Record<string, number>, color: 'white' | 'black'): string {
  const items: string[] = [];
  for (const type of POCKET_PIECE_ORDER) {
    const count = pocket[type] ?? 0;
    if (count <= 0) continue;
    const key = color === 'white' ? type : type.toLowerCase();
    const src = PIECE_IMAGES_SRC[key];
    if (!src) continue;
    items.push(`
      <div style="display:flex; align-items:center; gap:4px;">
        <img src="${src}" style="width:${POCKET_PIECE_SIZE}px; height:${POCKET_PIECE_SIZE}px;" />
        <span style="color:#3a2a1a; font-weight:700; font-size:14px;">×${count}</span>
      </div>
    `);
  }
  if (items.length === 0) {
    return `<div style="color:#7a6a55; font-style:italic; font-size:0.85rem;">empty</div>`;
  }
  return items.join('');
}

export function renderer(options: RendererOptions) {
  const { replay, step, parent } = options;
  const steps = (replay as any).steps as any[];
  const playerNames = (replay as any).info?.TeamNames || [];
  const width = parent.clientWidth || 400;
  const height = parent.clientHeight || 400;

  let currentRendererContainer: HTMLElement | null = null;
  let currentBoardContainer: HTMLElement | null = null;
  let currentBoardElement: HTMLElement | null = null;
  let blackPocketElement: HTMLElement | null = null;
  let whitePocketElement: HTMLElement | null = null;
  let statusTextElement: HTMLParagraphElement | null = null;
  let winnerTextElement: HTMLParagraphElement | null = null;
  let squareSize = 0;

  function _clearState(parentElementToClear: HTMLElement) {
    if (!parentElementToClear) return false;
    parentElementToClear.innerHTML = '';
    return true;
  }

  function makePocketBox(label: string): HTMLElement {
    const box = document.createElement('div');
    Object.assign(box.style, {
      backgroundColor: POCKET_BG,
      border: `2px solid ${POCKET_BORDER}`,
      borderRadius: '8px',
      padding: '8px 10px',
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
      height: '120px',
    });
    const heading = document.createElement('div');
    heading.textContent = label;
    Object.assign(heading.style, {
      color: '#3a2a1a',
      fontSize: '0.8rem',
      fontWeight: '700',
      letterSpacing: '0.04em',
      textTransform: 'uppercase',
    });
    const items = document.createElement('div');
    items.className = 'pocket-items';
    Object.assign(items.style, {
      display: 'flex',
      flexWrap: 'wrap',
      alignItems: 'center',
      gap: '8px 12px',
    });
    box.appendChild(heading);
    box.appendChild(items);
    return box;
  }

  function _buildVisualizer(parentElement: HTMLElement, rows: number, cols: number) {
    const isMobile = window.innerWidth < 768;

    currentRendererContainer = document.createElement('div');
    Object.assign(currentRendererContainer.style, {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: isMobile ? '10px' : '20px',
      boxSizing: 'border-box',
      width: '100%',
      height: '100%',
      backgroundColor: '#202124',
    });
    parentElement.appendChild(currentRendererContainer);

    // Header: White player on the left, title in middle, Black player on the right.
    // OpenSpiel's crazyhouse maps player 0 -> Black and player 1 -> White, so
    // playerNames[1] is White and playerNames[0] is Black.
    const headerContainer = document.createElement('div');
    Object.assign(headerContainer.style, {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      width: '100%',
      marginBottom: '0.75rem',
      color: 'white',
      flexShrink: '0',
      flexDirection: isMobile ? 'column' : 'row',
    });
    const whitePlayerContainer = document.createElement('div');
    Object.assign(whitePlayerContainer.style, { display: 'flex', alignItems: 'center' });
    const whitePawn = document.createElement('img');
    whitePawn.src = PIECE_IMAGES_SRC.P;
    Object.assign(whitePawn.style, { height: '30px', marginRight: '8px' });
    const whiteName = document.createElement('span');
    whiteName.textContent = playerNames.length > 1 ? playerNames[1] : 'White';
    Object.assign(whiteName.style, { fontSize: isMobile ? '1rem' : '1.1rem', fontWeight: 'bold' });
    whitePlayerContainer.appendChild(whitePawn);
    whitePlayerContainer.appendChild(whiteName);

    const titleElement = document.createElement('h1');
    titleElement.textContent = 'Crazyhouse';
    Object.assign(titleElement.style, {
      fontSize: isMobile ? '1.3rem' : '1.6rem',
      fontWeight: 'bold',
      textAlign: 'center',
      color: '#e5e7eb',
      margin: isMobile ? '8px 0' : '0 32px',
    });

    const blackPlayerContainer = document.createElement('div');
    Object.assign(blackPlayerContainer.style, { display: 'flex', alignItems: 'center' });
    const blackName = document.createElement('span');
    blackName.textContent = playerNames.length > 0 ? playerNames[0] : 'Black';
    Object.assign(blackName.style, { fontSize: isMobile ? '1rem' : '1.1rem', fontWeight: 'bold' });
    const blackPawn = document.createElement('img');
    blackPawn.src = PIECE_IMAGES_SRC.p;
    Object.assign(blackPawn.style, { height: '30px', marginLeft: '8px' });
    blackPlayerContainer.appendChild(blackName);
    blackPlayerContainer.appendChild(blackPawn);

    headerContainer.appendChild(whitePlayerContainer);
    headerContainer.appendChild(titleElement);
    headerContainer.appendChild(blackPlayerContainer);
    currentRendererContainer.appendChild(headerContainer);

    // Board + side pocket panel laid out horizontally.
    const boardRow = document.createElement('div');
    Object.assign(boardRow.style, {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'stretch',
      gap: '16px',
      flexGrow: '1',
      width: '100%',
      minHeight: '0',
    });
    currentRendererContainer.appendChild(boardRow);

    currentBoardContainer = document.createElement('div');
    Object.assign(currentBoardContainer.style, {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      flexGrow: '1',
      minWidth: '0',
      minHeight: '0',
    });
    boardRow.appendChild(currentBoardContainer);

    currentBoardElement = document.createElement('div');
    Object.assign(currentBoardElement.style, {
      display: 'grid',
      border: '2px solid #333',
    });
    currentBoardContainer.appendChild(currentBoardElement);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const square = document.createElement('div');
        square.id = `cell-${r}-${c}`;
        Object.assign(square.style, {
          backgroundColor: (r + c) % 2 === 0 ? LIGHT_SQUARE_COLOR : DARK_SQUARE_COLOR,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        });
        currentBoardElement.appendChild(square);
      }
    }

    // Side panel: pockets stacked, with Black on top (matches board orientation
    // where rank 8 / Black is at the top).
    const pocketPanel = document.createElement('div');
    Object.assign(pocketPanel.style, {
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-evenly',
      gap: '12px',
      flexShrink: '0',
      width: isMobile ? '120px' : '160px',
    });
    boardRow.appendChild(pocketPanel);

    blackPocketElement = makePocketBox('Black pocket');
    whitePocketElement = makePocketBox('White pocket');
    pocketPanel.appendChild(blackPocketElement);
    pocketPanel.appendChild(whitePocketElement);

    // Status panel.
    const statusContainer = document.createElement('div');
    Object.assign(statusContainer.style, {
      padding: '6px 12px',
      backgroundColor: 'white',
      borderRadius: '8px',
      boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)',
      textAlign: 'center',
      minWidth: '200px',
      maxWidth: '90vw',
      marginTop: '10px',
      flexShrink: '0',
    });
    currentRendererContainer.appendChild(statusContainer);

    statusTextElement = document.createElement('p');
    Object.assign(statusTextElement.style, {
      fontSize: isMobile ? '0.85rem' : '1rem',
      fontWeight: '600',
      margin: '0 0 4px 0',
    });
    statusContainer.appendChild(statusTextElement);

    winnerTextElement = document.createElement('p');
    Object.assign(winnerTextElement.style, {
      fontSize: isMobile ? '0.9rem' : '1.05rem',
      fontWeight: '700',
      margin: '0',
    });
    statusContainer.appendChild(winnerTextElement);

    return true;
  }

  function _renderBoard(obs: CrazyhouseObservation | null, displayRows: number, displayCols: number) {
    if (
      !currentBoardContainer ||
      !currentBoardElement ||
      !blackPocketElement ||
      !whitePocketElement ||
      !statusTextElement ||
      !winnerTextElement ||
      !obs
    ) {
      return;
    }

    const containerWidth = currentBoardContainer.clientWidth || width;
    const containerHeight = currentBoardContainer.clientHeight || height;
    let smallestEdge = Math.min(containerWidth, containerHeight);
    smallestEdge = smallestEdge > 200 ? smallestEdge - 24 : smallestEdge;
    const newSquareSize = Math.floor(smallestEdge / displayCols);

    if (newSquareSize !== squareSize) {
      squareSize = newSquareSize;
      Object.assign(currentBoardElement.style, {
        gridTemplateColumns: `repeat(${displayCols}, ${squareSize}px)`,
        gridTemplateRows: `repeat(${displayRows}, ${squareSize}px)`,
        width: `${displayCols * squareSize}px`,
        height: `${displayRows * squareSize}px`,
      });
      currentBoardElement.querySelectorAll('div[id^="cell-"]').forEach((square) => {
        const div = square as HTMLDivElement;
        div.style.width = `${squareSize}px`;
        div.style.height = `${squareSize}px`;
      });
    }

    // Pieces.
    for (let r = 0; r < displayRows; r++) {
      for (let c = 0; c < displayCols; c++) {
        const cell = currentBoardElement.querySelector(`#cell-${r}-${c}`);
        if (cell) cell.innerHTML = '';
      }
    }
    const pieceSize = Math.floor(squareSize * 0.9);
    for (let r = 0; r < displayRows; r++) {
      const row = obs.board[r] ?? [];
      for (let c = 0; c < displayCols; c++) {
        const piece = row[c];
        const cell = currentBoardElement.querySelector(`#cell-${r}-${c}`);
        if (!cell || !piece || piece === '.') continue;
        const src = PIECE_IMAGES_SRC[piece];
        if (!src) continue;
        const img = document.createElement('img');
        img.src = src;
        img.style.width = `${pieceSize}px`;
        img.style.height = `${pieceSize}px`;
        cell.appendChild(img);
      }
    }

    // Pockets — write into the .pocket-items slot inside each box.
    const blackItems = blackPocketElement.querySelector('.pocket-items') as HTMLElement | null;
    const whiteItems = whitePocketElement.querySelector('.pocket-items') as HTMLElement | null;
    if (blackItems) blackItems.innerHTML = buildPocketHtml(obs.pockets?.black ?? {}, 'black');
    if (whiteItems) whiteItems.innerHTML = buildPocketHtml(obs.pockets?.white ?? {}, 'white');

    // Status.
    statusTextElement.innerHTML = '';
    winnerTextElement.innerHTML = '';
    if (obs.is_terminal) {
      const winnerLabel =
        obs.winner === 'draw'
          ? 'Draw'
          : obs.winner
            ? `🎉 ${obs.winner[0].toUpperCase()}${obs.winner.slice(1)} wins!`
            : 'Game over';
      statusTextElement.textContent = 'Final';
      winnerTextElement.innerHTML = `<span style="color:black;">${winnerLabel}</span>`;
    } else {
      const sideLabel = obs.side_to_move === 'w' ? 'White' : 'Black';
      const playerName =
        typeof obs.current_player === 'string' && playerNames.length > 1
          ? obs.current_player === 'white'
            ? playerNames[1]
            : playerNames[0]
          : null;
      const turnText = playerName ? `${sideLabel} (${playerName})` : sideLabel;
      statusTextElement.innerHTML = `<span style="color:black;">Move ${obs.fullmove_number} — ${turnText} to move</span>`;
    }
  }

  if (!_clearState(parent)) {
    parent.innerHTML = "<p style='color:red; font-family: sans-serif;'>Renderer setup failed.</p>";
    return;
  }

  _buildVisualizer(parent, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);

  const obs = steps && steps.length > step ? parseObservation(steps[step]) : null;

  // Defer rendering to next frame so the flex container has a measured size.
  requestAnimationFrame(() => {
    _renderBoard(obs, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
  });
}
