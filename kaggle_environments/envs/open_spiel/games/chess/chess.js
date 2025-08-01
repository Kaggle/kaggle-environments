function renderer(options) {
    const { environment, step, parent, width = 400, height = 400 } = options;

    // Chess-specific constants
    const DEFAULT_NUM_ROWS = 8;
    const DEFAULT_NUM_COLS = 8;
    const PIECE_SVG_URLS = {
        'p': 'https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg', // Black Pawn
        'r': 'https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg', // Black Rook
        'n': 'https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg', // Black Knight
        'b': 'https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg', // Black Bishop
        'q': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg', // Black Queen
        'k': 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg', // Black King
        'P': 'https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg', // White Pawn
        'R': 'https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg', // White Rook
        'N': 'https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg', // White Knight
        'B': 'https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg', // White Bishop
        'Q': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg', // White Queen
        'K': 'https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg'  // White King
    };
    const LIGHT_SQUARE_COLOR = '#f0d9b5';
    const DARK_SQUARE_COLOR = '#b58863';

    // Renderer state variables
    let currentBoardElement = null;
    let currentStatusTextElement = null;
    let currentWinnerTextElement = null;
    let currentMessageBoxElement = typeof document !== 'undefined' ? document.getElementById('messageBox') : null;
    let currentRendererContainer = null;
    let currentTitleElement = null;
    let squareSize = 50;

    function _showMessage(message, type = 'info', duration = 3000) {
        if (typeof document === 'undefined' || !document.body) return;
        if (!currentMessageBoxElement) {
            currentMessageBoxElement = document.createElement('div');
            currentMessageBoxElement.id = 'messageBox';
            // Identical styling to the Connect Four renderer's message box
            Object.assign(currentMessageBoxElement.style, {
                position: 'fixed',
                top: '10px',
                left: '50%',
                transform: 'translateX(-50%)',
                padding: '0.75rem 1rem',
                borderRadius: '0.375rem',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                zIndex: '1000',
                opacity: '0',
                transition: 'opacity 0.3s ease-in-out, background-color 0.3s',
                fontSize: '0.875rem',
                fontFamily: "'Inter', sans-serif",
                color: 'white'
            });
            document.body.appendChild(currentMessageBoxElement);
        }
        currentMessageBoxElement.textContent = message;
        currentMessageBoxElement.style.backgroundColor = type === 'error' ? '#ef4444' : '#10b981';
        currentMessageBoxElement.style.opacity = '1';
        setTimeout(() => { if (currentMessageBoxElement) currentMessageBoxElement.style.opacity = '0'; }, duration);
    }

    function _ensureRendererElements(parentElementToClear, rows, cols) {
        if (!parentElementToClear) return false;
        parentElementToClear.innerHTML = '';

        currentRendererContainer = document.createElement('div');
        Object.assign(currentRendererContainer.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            padding: '20px',
            boxSizing: 'border-box',
            width: '100%',
            height: '100%',
            fontFamily: "'Inter', sans-serif"
        });

        if (!environment.viewer) {
          currentTitleElement = document.createElement('h1');
          currentTitleElement.textContent = 'Chess';
          // Identical styling to the Connect Four renderer's title
          Object.assign(currentTitleElement.style, {
            fontSize: '1.875rem',
            fontWeight: 'bold',
            marginBottom: '1rem',
            textAlign: 'center',
            color: '#2563eb'
          });
          currentRendererContainer.appendChild(currentTitleElement);
        }
        parentElementToClear.appendChild(currentRendererContainer);

        const smallestParentEdge = Math.min(width, height);
        squareSize = Math.floor(smallestParentEdge / DEFAULT_NUM_COLS);
        currentBoardElement = document.createElement('div');
        Object.assign(currentBoardElement.style, {
            display: 'grid',
            gridTemplateColumns: `repeat(${cols}, ${squareSize}px)`,
            gridTemplateRows: `repeat(${rows}, ${squareSize}px)`,
            width: `${cols * squareSize}px`,
            height: `${rows * squareSize}px`,
            border: '2px solid #333'
        });

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const square = document.createElement('div');
                square.id = `cell-${r}-${c}`;
                Object.assign(square.style, {
                    width: `${squareSize}px`,
                    height: `${squareSize}px`,
                    backgroundColor: (r + c) % 2 === 0 ? LIGHT_SQUARE_COLOR : DARK_SQUARE_COLOR,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                });
                currentBoardElement.appendChild(square);
            }
        }
        currentRendererContainer.appendChild(currentBoardElement);

        const statusContainer = document.createElement('div');
        // Identical styling to the Connect Four renderer's status container
        Object.assign(statusContainer.style, {
            padding: '10px 15px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)',
            textAlign: 'center',
            width: 'auto',
            minWidth: '200px',
            maxWidth: '90vw',
            marginTop: '20px'
        });
        if (!environment.viewer) {
          currentRendererContainer.appendChild(statusContainer);
        }

        currentStatusTextElement = document.createElement('p');
        Object.assign(currentStatusTextElement.style, {
            fontSize: '1.1rem',
            fontWeight: '600',
            margin: '0 0 5px 0'
        });
        statusContainer.appendChild(currentStatusTextElement);

        currentWinnerTextElement = document.createElement('p');
        Object.assign(currentWinnerTextElement.style, {
            fontSize: '1.25rem',
            fontWeight: '700',
            margin: '5px 0 0 0'
        });
        statusContainer.appendChild(currentWinnerTextElement);

        if (typeof document !== 'undefined' && !document.body.hasAttribute('data-renderer-initialized')) {
            document.body.setAttribute('data-renderer-initialized', 'true');
        }
        return true;
    }

    // White is in position 1, Black is in position 0
    function _getTeamNameForColor(color, teamNames) {
        if (!teamNames || teamNames.length < 2) return null;
        return color.toLowerCase() === 'white' ? teamNames[1] : teamNames[0];
    }

    function _deriveWinnerFromRewards(currentStepAgents, teamNames) {
        if (!currentStepAgents || currentStepAgents.length < 2) return null;
        
        const player0Reward = currentStepAgents[0].reward;
        const player1Reward = currentStepAgents[1].reward;
        
        if (player0Reward === player1Reward) {
            return 'draw';
        }
        
        const winnerPlayerIndex = player0Reward === 1.0 ? 0 : 1;
        const color = winnerPlayerIndex === 0 ? 'Black' : 'White';
        
        if (teamNames) {
            const teamName = _getTeamNameForColor(color, teamNames);
            return `${color} (${teamName})`;
        }
        
        return color.toLowerCase();
    }

    function _parseFen(fen) {
        if (!fen || typeof fen !== 'string') return null;

        const [piecePlacement, activeColor, castling, enPassant, halfmoveClock, fullmoveNumber] = fen.split(' ');
        const board = [];
        const rows = piecePlacement.split('/');

        for (const row of rows) {
            const boardRow = [];
            for (const char of row) {
                if (isNaN(parseInt(char))) {
                    boardRow.push(char);
                } else {
                    for (let i = 0; i < parseInt(char); i++) {
                        boardRow.push(null);
                    }
                }
            }
            board.push(boardRow);
        }

        return {
            board,
            activeColor,
            castling,
            enPassant,
            halfmoveClock,
            fullmoveNumber
        };
    }


    function _renderBoardDisplay(gameStateToDisplay, displayRows, displayCols) {
        if (!currentBoardElement || !currentStatusTextElement || !currentWinnerTextElement) return;

        // Clear board
        for (let r = 0; r < displayRows; r++) {
            for (let c = 0; c < displayCols; c++) {
                const squareElement = currentBoardElement.querySelector(`#cell-${r}-${c}`);
                if (squareElement) {
                    squareElement.innerHTML = '';
                }
            }
        }
        
        if (!gameStateToDisplay || !gameStateToDisplay.board) {
            currentStatusTextElement.textContent = "Waiting for game data...";
            currentWinnerTextElement.textContent = "";
            return;
        }


        const { board, activeColor, isTerminal, winner } = gameStateToDisplay;

        const pieceSize = Math.floor(squareSize * 0.9);
        for (let r_data = 0; r_data < displayRows; r_data++) {
            for (let c_data = 0; c_data < displayCols; c_data++) {
                const piece = board[r_data][c_data];
                const squareElement = currentBoardElement.querySelector(`#cell-${r_data}-${c_data}`);
                if (squareElement && piece) {
                    const pieceImg = document.createElement('img');
                    pieceImg.src = PIECE_SVG_URLS[piece];
                    pieceImg.style.width = `${pieceSize}px`;
                    pieceImg.style.height = `${pieceSize}px`;
                    squareElement.appendChild(pieceImg);
                }
            }
        }

        currentStatusTextElement.innerHTML = '';
        currentWinnerTextElement.innerHTML = '';
        if (isTerminal) {
            currentStatusTextElement.textContent = "Game Over!";
            if (winner) {
                 if (String(winner).toLowerCase() === 'draw') {
                    currentWinnerTextElement.textContent = "It's a Draw!";
                 } else {
                    currentWinnerTextElement.innerHTML = `Winner: <span style="font-weight: bold;">${winner}</span>`;
                 }
            } else {
                 currentWinnerTextElement.textContent = "Game ended.";
            }
        } else {
          const playerColor = String(activeColor).toLowerCase() === 'w' ? 'White' : 'Black';
          const teamName = _getTeamNameForColor(playerColor, environment.info?.TeamNames);
          const currentPlayerText = teamName ? `${playerColor} (${teamName})` : playerColor;
          
          currentStatusTextElement.innerHTML = `Current Player: <span style="font-weight: bold;">${currentPlayerText}</span>`;
        }
    }

    // --- Main execution logic ---
    if (!_ensureRendererElements(parent, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS)) {
        if (parent && typeof parent.innerHTML !== 'undefined') {
            parent.innerHTML = "<p style='color:red; font-family: sans-serif;'>Critical Error: Renderer element setup failed.</p>";
        }
        return;
    }

    if (!environment || !environment.steps || !environment.steps[step]) {
        _renderBoardDisplay(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if (currentStatusTextElement) currentStatusTextElement.textContent = "Initializing environment...";
        return;
    }

    const currentStepAgents = environment.steps[step];
    if (!currentStepAgents || !Array.isArray(currentStepAgents) || currentStepAgents.length === 0) {
        _renderBoardDisplay(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if (currentStatusTextElement) currentStatusTextElement.textContent = "Waiting for agent data...";
        return;
    }
    
    // In chess, observation is the same for both agents. We can take it from the first.
    const agent = currentStepAgents[0];

    if (!agent || typeof agent.observation === 'undefined') {
        _renderBoardDisplay(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if (currentStatusTextElement) currentStatusTextElement.textContent = "Waiting for observation data...";
        return;
    }
    const observationForRenderer = agent.observation;

    let gameSpecificState = null;

    if (observationForRenderer && typeof observationForRenderer.observationString === 'string' && observationForRenderer.observationString.trim() !== '') {
        try {
            const fen = observationForRenderer.observationString;
            const parsedFen = _parseFen(fen);
            if (parsedFen) {
                const winner = observationForRenderer.isTerminal ? 
                    _deriveWinnerFromRewards(currentStepAgents, environment.info?.TeamNames) : null;
                gameSpecificState = { 
                    ...parsedFen,
                    isTerminal: observationForRenderer.isTerminal,
                    winner: winner
                };
            }
        } catch (e) {
            _showMessage("Error: Corrupted game state (obs_string).", 'error');
        }
    }

    _renderBoardDisplay(gameSpecificState, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
}
