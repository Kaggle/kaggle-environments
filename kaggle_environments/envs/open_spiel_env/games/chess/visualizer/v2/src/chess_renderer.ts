import { ChessStep } from '@kaggle-environments/core';
import { DARK_SQUARE_COLOR, DEFAULT_NUM_COLS, DEFAULT_NUM_ROWS, LIGHT_SQUARE_COLOR, PIECE_IMAGES_SRC } from './consts';

export function renderer(options: any) {
  const { steps, step, parent, playerNames, width = 400, height = 400, viewer } = options;

  let currentBoardElement: HTMLElement | null = null;
  let currentStatusTextElement: HTMLParagraphElement | null = null;
  let currentWinnerTextElement: HTMLElement | null = null;
  let currentRendererContainer: HTMLElement | null = null;
  let currentBoardContainer: HTMLElement | null = null;
  let currentTitleElement: HTMLElement | null = null;
  let squareSize = 0;

  /* We need to clear the board and redraw it every time
  because we are appending elements to the document and
  they could get stale. */
  function _clearState(parentElementToClear: HTMLElement) {
    if (!parentElementToClear) return false;
    parentElementToClear.innerHTML = '';
    return true;
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
      fontFamily: "'Inter', sans-serif",
    });
    parentElement.appendChild(currentRendererContainer);

    if (!viewer) {
      const headerContainer = document.createElement('div');
      Object.assign(headerContainer.style, {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        width: '100%',
        marginBottom: '1rem',
        color: 'white',
        flexShrink: '0',
        flexDirection: isMobile ? 'column' : 'row', // Stacks header vertically on mobile
      });

      // Player 2 (White) - Left side
      const whitePlayerContainer = document.createElement('div');
      Object.assign(whitePlayerContainer.style, {
        display: 'flex',
        alignItems: 'center',
      });
      const whitePawnImg = document.createElement('img');
      whitePawnImg.src = PIECE_IMAGES_SRC.P;
      Object.assign(whitePawnImg.style, { height: '30px', marginRight: '8px' });
      const whitePlayerName = document.createElement('span');
      whitePlayerName.textContent = playerNames.length > 1 ? playerNames[1] : 'White';
      Object.assign(whitePlayerName.style, {
        fontSize: isMobile ? '1rem' : '1.1rem',
        fontWeight: 'bold',
      });
      whitePlayerContainer.appendChild(whitePawnImg);
      whitePlayerContainer.appendChild(whitePlayerName);

      // Center Title
      currentTitleElement = document.createElement('h1');
      currentTitleElement.textContent = 'Chess';
      Object.assign(currentTitleElement.style, {
        fontSize: isMobile ? '1.5rem' : '1.875rem',
        fontWeight: 'bold',
        textAlign: 'center',
        color: '#e5e7eb',
        margin: isMobile ? '10px 0' : '0 40px',
        order: isMobile ? '0' : 'initial', // Ensures title is between players on desktop
      });

      // Player 1 (Black) - Right side
      const blackPlayerContainer = document.createElement('div');
      Object.assign(blackPlayerContainer.style, {
        display: 'flex',
        alignItems: 'center',
      });
      const blackPlayerName = document.createElement('span');
      blackPlayerName.textContent = playerNames.length > 1 ? playerNames[0] : 'Black';
      Object.assign(blackPlayerName.style, {
        fontSize: isMobile ? '1rem' : '1.1rem',
        fontWeight: 'bold',
      });
      const blackPawnImg = document.createElement('img');
      blackPawnImg.src = PIECE_IMAGES_SRC.p;
      Object.assign(blackPawnImg.style, { height: '30px', marginLeft: '8px' });
      blackPlayerContainer.appendChild(blackPlayerName);
      blackPlayerContainer.appendChild(blackPawnImg);

      // Assemble the header - order matters for mobile stacking
      if (isMobile) {
        // On mobile: White Player, then Black Player, then Title for a "vs" feel
        headerContainer.appendChild(whitePlayerContainer);
        const vsText = document.createElement('div');
        vsText.textContent = 'vs';
        Object.assign(vsText.style, { margin: '4px 0', fontStyle: 'italic' });
        headerContainer.appendChild(vsText);
        headerContainer.appendChild(blackPlayerContainer);
        // Title is not added here for mobile to keep it separate or could be added last
      } else {
        // On desktop: White Player, Title, Black Player
        headerContainer.appendChild(whitePlayerContainer);
        headerContainer.appendChild(currentTitleElement);
        headerContainer.appendChild(blackPlayerContainer);
      }
      currentRendererContainer.appendChild(headerContainer);
    }

    // ... (The rest of the function for the board and status container remains the same, but let's add responsive fonts to the status)

    // Board container...
    currentBoardContainer = document.createElement('div');
    Object.assign(currentBoardContainer.style, {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      flexGrow: '1',
      width: '100%',
      minHeight: '0',
    });
    currentRendererContainer.appendChild(currentBoardContainer);

    // ... code to create board and squares ...
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

    // Status Container
    const statusContainer = document.createElement('div');
    Object.assign(statusContainer.style, {
      padding: '5px',
      backgroundColor: 'white',
      borderRadius: '8px',
      boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)',
      textAlign: 'center',
      width: 'auto',
      minWidth: '200px',
      maxWidth: '90vw',
      marginTop: '10px',
      flexShrink: '0',
    });
    if (!viewer) {
      currentRendererContainer.appendChild(statusContainer);
    }

    currentStatusTextElement = document.createElement('p');
    Object.assign(currentStatusTextElement.style, {
      fontSize: isMobile ? '0.8rem' : '1.1rem',
      fontWeight: '600',
      margin: '0 0 5px 0',
    });
    statusContainer.appendChild(currentStatusTextElement);

    currentWinnerTextElement = document.createElement('p');
    Object.assign(currentWinnerTextElement.style, {
      fontSize: isMobile ? '0.9rem' : '1.1rem',
      fontWeight: '700',
      margin: '5px 0 0 0',
    });
    statusContainer.appendChild(currentWinnerTextElement);

    return true;
  }

  function _renderChessBoard(chessStep: ChessStep | null, displayRows: number, displayCols: number) {
    if (
      !currentBoardContainer ||
      !currentBoardElement ||
      !currentStatusTextElement ||
      !currentWinnerTextElement ||
      !chessStep
    ) {
      return;
    }

    const isMobile = window.innerWidth < 768;

    // Calculate and apply board size
    const containerWidth = currentBoardContainer?.clientWidth ?? width;
    const containerHeight = currentBoardContainer?.clientHeight ?? height;
    let smallestContainerEdge = Math.min(containerWidth, containerHeight);
    // This is greedily trying to take as much space as possible, which can cause some conflict with flex box calculations for other elements
    // we are going to take 24px off (arbitrary) to give the flex box renderer a bit of space to work with. Without it we will get some clipping.
    smallestContainerEdge = smallestContainerEdge > 200 ? smallestContainerEdge - 24 : smallestContainerEdge;
    const newSquareSize = Math.floor(smallestContainerEdge / displayCols);

    if (newSquareSize !== squareSize) {
      squareSize = newSquareSize;
      Object.assign(currentBoardElement.style, {
        gridTemplateColumns: `repeat(${displayCols}, ${squareSize}px)`,
        gridTemplateRows: `repeat(${displayRows}, ${squareSize}px)`,
        width: `${displayCols * squareSize}px`,
        height: `${displayRows * squareSize}px`,
      });

      const squares: NodeListOf<Element> = currentBoardElement.querySelectorAll('div[id^="cell-"]');
      squares.forEach((square: Element) => {
        const squareDiv = square as HTMLDivElement;
        Object.assign(squareDiv.style, {
          width: `${squareSize}px`,
          height: `${squareSize}px`,
        });
      });
    }

    // Clear and render board pieces...
    for (let r = 0; r < displayRows; r++) {
      for (let c = 0; c < displayCols; c++) {
        const squareElement = currentBoardElement.querySelector(`#cell-${r}-${c}`);
        if (squareElement) squareElement.innerHTML = '';
      }
    }

    const { board, activeColor } = chessStep.fenState;

    const pieceSize = Math.floor(squareSize * 0.9);
    for (let r_data = 0; r_data < displayRows; r_data++) {
      for (let c_data = 0; c_data < displayCols; c_data++) {
        const piece = board[r_data][c_data];
        const squareElement = currentBoardElement.querySelector(`#cell-${r_data}-${c_data}`);
        if (squareElement && piece) {
          const pieceImg = document.createElement('img');
          pieceImg.src = PIECE_IMAGES_SRC[piece];
          pieceImg.style.width = `${pieceSize}px`;
          pieceImg.style.height = `${pieceSize}px`;
          squareElement.appendChild(pieceImg);
        }
      }
    }

    // Render status text
    currentStatusTextElement.innerHTML = '';
    currentWinnerTextElement.innerHTML = '';
    if (chessStep.isTerminal) {
      if (isMobile) {
        currentStatusTextElement.innerHTML = `<div style="font-size: 0.9rem; color: #666;">Winner</div>`;
        currentWinnerTextElement.innerHTML = `<div style="font-size: 1.1rem; font-weight: bold;">${chessStep.winner}</div>`;
      } else {
        currentStatusTextElement.textContent = '';
        currentWinnerTextElement.innerHTML = `<span style="font-weight: bold; color: black;">${chessStep.winner}</span>`;
      }
    } else {
      const currentPlayer = chessStep.players.find((player) => player.isTurn);
      const currentPlayerText = currentPlayer ? `${activeColor} (${currentPlayer.name})` : activeColor;

      if (isMobile) {
        currentStatusTextElement.innerHTML = `<div style="font-size: 0.9rem; color: #666;">Current Player</div>`;
        currentWinnerTextElement.innerHTML = `<div style="font-size: 1.1rem; font-weight: bold;">${currentPlayerText}</div>`;
      } else {
        currentStatusTextElement.innerHTML = `<span style="font-weight: bold; color: black;">Current Player: ${currentPlayerText}</span>`;
        currentWinnerTextElement.innerHTML = ''; // Clear winner text when game is active
      }
    }
  }

  if (!_clearState(parent)) {
    if (parent && typeof parent.innerHTML !== 'undefined') {
      parent.innerHTML =
        "<p style='color:red; font-family: sans-serif;'>Critical Error: Renderer element setup failed.</p>";
    }
    return;
  }

  _buildVisualizer(parent, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);

  if (!steps || steps.length === 0 || steps.length < step) {
    _renderChessBoard(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
    return;
  }

  const currentStep: ChessStep = steps[step];

  if (!currentStep) {
    _renderChessBoard(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
    return;
  }

  _renderChessBoard(currentStep, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
}
