function renderer(options) {
    const { environment, step, parent, width = 400, height = 400 } = options; // Chess-specific constants

    const DEFAULT_NUM_ROWS = 8;
    const DEFAULT_NUM_COLS = 8;
    const PIECE_SVG_URLS = {
        p: 'https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg', // Black Pawn
        r: 'https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg', // Black Rook
        n: 'https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg', // Black Knight
        b: 'https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg', // Black Bishop
        q: 'https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg', // Black Queen
        k: 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg', // Black King
        P: 'https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg', // White Pawn
        R: 'https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg', // White Rook
        N: 'https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg', // White Knight
        B: 'https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg', // White Bishop
        Q: 'https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg', // White Queen
        K: 'https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg' // White King
    };
    const LIGHT_SQUARE_COLOR = '#f0d9b5';
    const DARK_SQUARE_COLOR = '#b58863';

    let currentBoardElement = null;
    let currentStatusTextElement = null;
    let currentWinnerTextElement = null;
    let currentMessageBoxElement = typeof document !== 'undefined' ? document.getElementById('messageBox') : null;
    let currentRendererContainer = null;
    let currentBoardContainer = null;
    let currentTitleElement = null;
    let resizeObserver = null;

    function _showMessage(message, type = 'info', duration = 3000) {
        if (typeof document === 'undefined' || !document.body) return;
        if (!currentMessageBoxElement) {
            currentMessageBoxElement = document.createElement('div');
            currentMessageBoxElement.id = 'messageBox';
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
        setTimeout(() => {
            if (currentMessageBoxElement) currentMessageBoxElement.style.opacity = '0';
        }, duration);
    }
    function _ensureRendererElements(parentElementToClear, rows, cols) {
        if (!parentElementToClear) return false;
        parentElementToClear.innerHTML = '';

        // NEW: Check for mobile screen size to apply responsive styles.
        const isMobile = window.innerWidth < 768;

        currentRendererContainer = document.createElement('div');
        Object.assign(currentRendererContainer.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            padding: isMobile ? '10px' : '20px', // Responsive padding
            boxSizing: 'border-box',
            width: '100%',
            height: '100%',
            fontFamily: "'Inter', sans-serif",
        });
        parentElementToClear.appendChild(currentRendererContainer);

        if (!environment.viewer) {
            const headerContainer = document.createElement('div');
            Object.assign(headerContainer.style, {
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%',
                marginBottom: '1rem',
                color: 'white',
                flexShrink: '0',
                flexDirection: isMobile ? 'column' : 'row' // Stacks header vertically on mobile
            });

            // Player 2 (White) - Left side
            const whitePlayerContainer = document.createElement('div');
            Object.assign(whitePlayerContainer.style, {
                display: 'flex',
                alignItems: 'center'
            });
            const whitePawnImg = document.createElement('img');
            whitePawnImg.src = PIECE_SVG_URLS.P;
            Object.assign(whitePawnImg.style, { height: '30px', marginRight: '8px' });
            const whitePlayerName = document.createElement('span');
            whitePlayerName.textContent = environment.info?.TeamNames?.[1] || 'Player 2';
            Object.assign(whitePlayerName.style, {
                fontSize: isMobile ? '1rem' : '1.1rem', // Responsive font size
                fontWeight: 'bold'
            });
            whitePlayerContainer.appendChild(whitePawnImg);
            whitePlayerContainer.appendChild(whitePlayerName);

            // Center Title
            currentTitleElement = document.createElement('h1');
            currentTitleElement.textContent = 'Chess';
            Object.assign(currentTitleElement.style, {
                fontSize: isMobile ? '1.5rem' : '1.875rem', // Responsive font size
                fontWeight: 'bold',
                textAlign: 'center',
                color: '#e5e7eb',
                margin: isMobile ? '10px 0' : '0 40px', // Responsive margin
                order: isMobile ? '0' : 'initial' // Ensures title is between players on desktop
            });

            // Player 1 (Black) - Right side
            const blackPlayerContainer = document.createElement('div');
            Object.assign(blackPlayerContainer.style, {
                display: 'flex',
                alignItems: 'center'
            });
            const blackPlayerName = document.createElement('span');
            blackPlayerName.textContent = environment.info?.TeamNames?.[0] || 'Player 1';
            Object.assign(blackPlayerName.style, {
                fontSize: isMobile ? '1rem' : '1.1rem', // Responsive font size
                fontWeight: 'bold'
            });
            const blackPawnImg = document.createElement('img');
            blackPawnImg.src = PIECE_SVG_URLS.p;
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
            overflow: 'hidden',
            width: '100%',
            minHeight: '0'
        });
        currentRendererContainer.appendChild(currentBoardContainer);

        // ... code to create board and squares ...
        currentBoardElement = document.createElement('div');
        Object.assign(currentBoardElement.style, {
            display: 'grid',
            border: '2px solid #333'
        });
        currentBoardContainer.appendChild(currentBoardElement);
        currentBoardElement = document.createElement('div');
        Object.assign(currentBoardElement.style, {
            display: 'grid',
            gridTemplateColumns: `repeat(${cols}, var(--square-size))`,
            gridTemplateRows: `repeat(${rows}, var(--square-size))`,
            border: '2px solid #333'
        });

        if (resizeObserver) {
          resizeObserver.disconnect();
        }
        currentBoardContainer.style.setProperty('--square-size', `${100/DEFAULT_NUM_COLS}%`);
        resizeObserver = new ResizeObserver(() => {
          const {width, height} = currentBoardContainer.getBoundingClientRect();
          const minSide = Math.min(width, height);
          currentBoardContainer.style.setProperty('--square-size', `${minSide/DEFAULT_NUM_COLS}px`);
        });
        resizeObserver.observe(currentBoardContainer)

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const square = document.createElement('div');
                square.id = `cell-${r}-${c}`;
                Object.assign(square.style, {
                    backgroundColor: (r + c) % 2 === 0 ? LIGHT_SQUARE_COLOR : DARK_SQUARE_COLOR,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                });
                currentBoardElement.appendChild(square);
            }
        }
        currentBoardContainer.appendChild(currentBoardElement);
        currentRendererContainer.appendChild(currentBoardContainer);

        // Status Container
        const statusContainer = document.createElement('div');
        Object.assign(statusContainer.style, {
            padding: '10px 15px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)',
            textAlign: 'center',
            width: 'auto',
            minWidth: '200px',
            maxWidth: '90vw',
            marginTop: '20px',
            flexShrink: '0'
        });
        if (!environment.viewer) {
            currentRendererContainer.appendChild(statusContainer);
        }

        currentStatusTextElement = document.createElement('p');
        Object.assign(currentStatusTextElement.style, {
            fontSize: isMobile ? '0.9rem' : '1.1rem', // Responsive font size
            fontWeight: '600',
            margin: '0 0 5px 0'
        });
        statusContainer.appendChild(currentStatusTextElement);

        currentWinnerTextElement = document.createElement('p');
        Object.assign(currentWinnerTextElement.style, {
            fontSize: isMobile ? '1rem' : '1.25rem', // Responsive font size
            fontWeight: '700',
            margin: '5px 0 0 0'
        });
        statusContainer.appendChild(currentWinnerTextElement);

        return true;
    }

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

        const winnerPlayerIndex = player0Reward === 1 ? 0 : 1;
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
        if (!currentBoardContainer || !currentBoardElement || !currentStatusTextElement || !currentWinnerTextElement)
            return;

        const isMobile = window.innerWidth < 768;

        // Calculate and apply board size
        const containerWidth = currentBoardContainer.clientWidth ?? width;
        const containerHeight = currentBoardContainer.clientHeight ?? height;
        const smallestContainerEdge = Math.min(containerWidth, containerHeight);
        const newSquareSize = Math.floor(smallestContainerEdge / displayCols);

        if (newSquareSize !== squareSize) {
            squareSize = newSquareSize;
            Object.assign(currentBoardElement.style, {
                gridTemplateColumns: `repeat(${displayCols}, ${squareSize}px)`,
                gridTemplateRows: `repeat(${displayRows}, ${squareSize}px)`,
                width: `${displayCols * squareSize}px`,
                height: `${displayRows * squareSize}px`
            });

            const squares = currentBoardElement.querySelectorAll('div[id^="cell-"]');
            squares.forEach((square) => {
                Object.assign(square.style, {
                    width: `${squareSize}px`,
                    height: `${squareSize}px`
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

        if (!gameStateToDisplay || !gameStateToDisplay.board) {
            currentStatusTextElement.textContent = 'Waiting for game data...';
            currentWinnerTextElement.textContent = '';
            return;
        }

        const { board, activeColor, isTerminal, winner } = gameStateToDisplay;
        for (let r_data = 0; r_data < displayRows; r_data++) {
            for (let c_data = 0; c_data < displayCols; c_data++) {
                const piece = board[r_data][c_data];
                const squareElement = currentBoardElement.querySelector(`#cell-${r_data}-${c_data}`);
                if (squareElement && piece) {
                    const pieceImg = document.createElement('img');
                    pieceImg.src = PIECE_SVG_URLS[piece];
                    pieceImg.style.width = `90%`;
                    pieceImg.style.height = `90%`;
                    squareElement.appendChild(pieceImg);
                }
            }
        }

        // Render status text
        currentStatusTextElement.innerHTML = '';
        currentWinnerTextElement.innerHTML = '';
        if (isTerminal) {
            const winnerText = winner
                ? String(winner).toLowerCase() === 'draw'
                    ? "It's a Draw!"
                    : winner
                : 'Game ended.';
            if (isMobile) {
                currentStatusTextElement.innerHTML = `<div style="font-size: 0.9rem; color: #666;">Winner</div>`;
                currentWinnerTextElement.innerHTML = `<div style="font-size: 1.1rem; font-weight: bold;">${winnerText}</div>`;
            } else {
                currentStatusTextElement.textContent = 'Game Over!';
                currentWinnerTextElement.innerHTML = `Winner: <span style="font-weight: bold;">${winnerText}</span>`;
            }
        } else {
            const playerColor = String(activeColor).toLowerCase() === 'w' ? 'White' : 'Black';
            const teamName = _getTeamNameForColor(playerColor, environment.info?.TeamNames);
            const currentPlayerText = teamName ? `${playerColor} (${teamName})` : playerColor;

            if (isMobile) {
                currentStatusTextElement.innerHTML = `<div style="font-size: 0.9rem; color: #666;">Current Player</div>`;
                currentWinnerTextElement.innerHTML = `<div style="font-size: 1.1rem; font-weight: bold;">${currentPlayerText}</div>`;
            } else {
                currentStatusTextElement.innerHTML = `Current Player: <span style="font-weight: bold;">${currentPlayerText}</span>`;
                currentWinnerTextElement.innerHTML = ''; // Clear winner text when game is active
            }
        }
    }

    if (!_ensureRendererElements(parent, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS)) {
        if (parent && typeof parent.innerHTML !== 'undefined') {
            parent.innerHTML =
                "<p style='color:red; font-family: sans-serif;'>Critical Error: Renderer element setup failed.</p>";
        }
        return;
    }

    if (!environment || !environment.steps || !environment.steps[step]) {
        _renderBoardDisplay(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if (currentStatusTextElement) currentStatusTextElement.textContent = 'Initializing environment...';
        return;
    }

    const currentStepAgents = environment.steps[step];
    if (!currentStepAgents || !Array.isArray(currentStepAgents) || currentStepAgents.length === 0) {
        _renderBoardDisplay(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if (currentStatusTextElement) currentStatusTextElement.textContent = 'Waiting for agent data...';
        return;
    }

    const agent = currentStepAgents[0];

    if (!agent || typeof agent.observation === 'undefined') {
        _renderBoardDisplay(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if (currentStatusTextElement) currentStatusTextElement.textContent = 'Waiting for observation data...';
        return;
    }
    const observationForRenderer = agent.observation;

    let gameSpecificState = null;

    if (
        observationForRenderer &&
        typeof observationForRenderer.observationString === 'string' &&
        observationForRenderer.observationString.trim() !== ''
    ) {
        try {
            const fen = observationForRenderer.observationString;
            const parsedFen = _parseFen(fen);
            if (parsedFen) {
                const winner = observationForRenderer.isTerminal
                    ? _deriveWinnerFromRewards(currentStepAgents, environment.info?.TeamNames)
                    : null;
                gameSpecificState = {
                    ...parsedFen,
                    isTerminal: observationForRenderer.isTerminal,
                    winner: winner
                };
            }
        } catch (e) {
            _showMessage('Error: Corrupted game state (obs_string).', 'error');
        }
    }

    _renderBoardDisplay(gameSpecificState, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
}
