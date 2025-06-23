function renderer(options) {
    const { environment, step, parent, interactive, isInteractive } = options;

    const DEFAULT_NUM_ROWS = 3;
    const DEFAULT_NUM_COLS = 3;
    const PLAYER_SYMBOLS = ['O', 'X'];
    const PLAYER_COLORS = ['#000000', '#000000'];
    const BOARD_BACKGROUND_COLOR = '#f0f0f0';
    const GRID_LINE_COLOR = '#cccccc';
    const MARK_COLOR = '#000000';
    const KAGGLE_BLUE = '#20BEFF';

    const SVG_NS = "http://www.w3.org/2000/svg";
    const CELL_UNIT_SIZE = 100;
    const MARK_THICKNESS = CELL_UNIT_SIZE * 0.08;
    const O_MARK_RADIUS = CELL_UNIT_SIZE * 0.32;
    const X_MARK_ARM_LENGTH = CELL_UNIT_SIZE * 0.30;

    const SVG_VIEWBOX_WIDTH = DEFAULT_NUM_COLS * CELL_UNIT_SIZE;
    const SVG_VIEWBOX_HEIGHT = DEFAULT_NUM_ROWS * CELL_UNIT_SIZE;

    let currentBoardSvgElement = null;
    let currentStatusTextElement = null;
    let currentWinnerTextElement = null;
    let currentMessageBoxElement = typeof document !== 'undefined' ? document.getElementById('messageBox') : null;
    let currentRendererContainer = null;
    let currentTitleElement = null;

    function _showMessage(message, type = 'info', duration = 3000) {
        if (typeof document === 'undefined' || !document.body) return;
        if (!currentMessageBoxElement) {
            currentMessageBoxElement = document.createElement('div');
            currentMessageBoxElement.id = 'messageBox';
            currentMessageBoxElement.style.position = 'fixed';
            currentMessageBoxElement.style.top = '20px';
            currentMessageBoxElement.style.left = '50%';
            currentMessageBoxElement.style.transform = 'translateX(-50%)';
            currentMessageBoxElement.style.padding = '0.75rem 1.25rem';
            currentMessageBoxElement.style.borderRadius = '0.375rem';
            currentMessageBoxElement.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)';
            currentMessageBoxElement.style.zIndex = '10000';
            currentMessageBoxElement.style.opacity = '0';
            currentMessageBoxElement.style.transition = 'opacity 0.3s ease-in-out, background-color 0.3s, top 0.3s';
            currentMessageBoxElement.style.fontSize = '0.9rem';
            currentMessageBoxElement.style.fontFamily = "'Inter', sans-serif";
            currentMessageBoxElement.style.maxWidth = '90%';
            currentMessageBoxElement.style.textAlign = 'center';
            document.body.appendChild(currentMessageBoxElement);
        }
        currentMessageBoxElement.textContent = message;
        currentMessageBoxElement.style.backgroundColor = type === 'error' ? '#ef4444' : '#10b981';
        currentMessageBoxElement.style.color = 'white';
        currentMessageBoxElement.style.opacity = '1';
        currentMessageBoxElement.style.top = '20px';
        setTimeout(() => {
            if (currentMessageBoxElement) {
                currentMessageBoxElement.style.opacity = '0';
            }
        }, duration);
    }

    function _ensureRendererElements(parentElementToClear, rows, cols) {
        if (!parentElementToClear) {
            console.error("Parent element to clear is null or undefined.");
            return false;
        }
        parentElementToClear.innerHTML = '';

        currentRendererContainer = document.createElement('div');
        currentRendererContainer.style.display = 'flex';
        currentRendererContainer.style.flexDirection = 'column';
        currentRendererContainer.style.alignItems = 'center';
        currentRendererContainer.style.padding = '20px';
        currentRendererContainer.style.boxSizing = 'border-box';
        currentRendererContainer.style.width = '100%';
        currentRendererContainer.style.height = '100%';
        currentRendererContainer.style.fontFamily = "'Inter', sans-serif";

        currentTitleElement = document.createElement('h1');
        currentTitleElement.textContent = 'Tic Tac Toe';
        currentTitleElement.style.fontSize = '1.875rem';
        currentTitleElement.style.fontWeight = 'bold';
        currentTitleElement.style.marginBottom = '1rem';
        currentTitleElement.style.textAlign = 'center';
        currentTitleElement.style.color = KAGGLE_BLUE;
        currentRendererContainer.appendChild(currentTitleElement);

        currentBoardSvgElement = document.createElementNS(SVG_NS, "svg");
        currentBoardSvgElement.setAttribute("viewBox", `0 0 ${SVG_VIEWBOX_WIDTH} ${SVG_VIEWBOX_HEIGHT}`);
        currentBoardSvgElement.setAttribute("preserveAspectRatio", "xMidYMid meet");
        currentBoardSvgElement.style.width = "auto";
        currentBoardSvgElement.style.maxWidth = "300px";
        currentBoardSvgElement.style.maxHeight = `calc(100vh - 280px)`;
        currentBoardSvgElement.style.aspectRatio = `${cols} / ${rows}`;
        currentBoardSvgElement.style.display = "block";
        currentBoardSvgElement.style.margin = "0 auto 20px auto";
        currentBoardSvgElement.style.backgroundColor = BOARD_BACKGROUND_COLOR;
        currentBoardSvgElement.style.borderRadius = "8px";

        for (let i = 1; i < rows; i++) {
            const line = document.createElementNS(SVG_NS, "line");
            line.setAttribute("x1", "0");
            line.setAttribute("y1", (i * CELL_UNIT_SIZE).toString());
            line.setAttribute("x2", SVG_VIEWBOX_WIDTH.toString());
            line.setAttribute("y2", (i * CELL_UNIT_SIZE).toString());
            line.setAttribute("stroke", GRID_LINE_COLOR);
            line.setAttribute("stroke-width", "2");
            currentBoardSvgElement.appendChild(line);
        }
        for (let i = 1; i < cols; i++) {
            const line = document.createElementNS(SVG_NS, "line");
            line.setAttribute("x1", (i * CELL_UNIT_SIZE).toString());
            line.setAttribute("y1", "0");
            line.setAttribute("x2", (i * CELL_UNIT_SIZE).toString());
            line.setAttribute("y2", SVG_VIEWBOX_HEIGHT.toString());
            line.setAttribute("stroke", GRID_LINE_COLOR);
            line.setAttribute("stroke-width", "2");
            currentBoardSvgElement.appendChild(line);
        }
        currentRendererContainer.appendChild(currentBoardSvgElement);

        const statusContainer = document.createElement('div');
        statusContainer.style.padding = '10px 15px';
        statusContainer.style.backgroundColor = 'white';
        statusContainer.style.borderRadius = '8px';
        statusContainer.style.boxShadow = '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)';
        statusContainer.style.textAlign = 'center';
        statusContainer.style.width = 'auto';
        statusContainer.style.minWidth = '220px';
        statusContainer.style.maxWidth = '90vw';
        currentRendererContainer.appendChild(statusContainer);

        currentStatusTextElement = document.createElement('p');
        currentStatusTextElement.style.fontSize = '1.1rem';
        currentStatusTextElement.style.fontWeight = '600';
        currentStatusTextElement.style.color = '#333333';
        currentStatusTextElement.style.margin = '0 0 5px 0';
        statusContainer.appendChild(currentStatusTextElement);

        currentWinnerTextElement = document.createElement('p');
        currentWinnerTextElement.style.fontSize = '1.25rem';
        currentWinnerTextElement.style.fontWeight = '700';
        currentWinnerTextElement.style.color = '#333333';
        currentWinnerTextElement.style.margin = '5px 0 0 0';
        statusContainer.appendChild(currentWinnerTextElement);

        parentElementToClear.appendChild(currentRendererContainer);

        if (typeof document !== 'undefined' && !document.body.hasAttribute('data-renderer-initialized')) {
            _showMessage("Renderer initialized (Tic Tac Toe).", "info", 1500);
            document.body.setAttribute('data-renderer-initialized', 'true');
        }
        return true;
    }

    function _renderBoardDisplay_svg(gameStateToDisplay, displayRows, displayCols) {
        if (!currentBoardSvgElement || !currentStatusTextElement || !currentWinnerTextElement) {
            console.error("Rendering elements not ready. This should not happen if _ensureRendererElements succeeded.");
            return;
        }

        const existingMarks = currentBoardSvgElement.querySelectorAll(".game-mark");
        existingMarks.forEach(mark => mark.remove());

        if (!gameStateToDisplay || typeof gameStateToDisplay.board !== 'object' || !Array.isArray(gameStateToDisplay.board) || gameStateToDisplay.board.length !== (displayRows * displayCols)) {
            currentStatusTextElement.textContent = "Waiting for player...";
            currentWinnerTextElement.textContent = "";
            return;
        }

        const { board, current_player, is_terminal, winner } = gameStateToDisplay;

        for (let i = 0; i < board.length; i++) {
            const cellValue = board[i];
            if (cellValue === null || cellValue === undefined || String(cellValue).trim() === '') {
                continue;
            }
            const row = Math.floor(i / displayCols);
            const col = i % displayCols;
            const cx = col * CELL_UNIT_SIZE + CELL_UNIT_SIZE / 2;
            const cy = row * CELL_UNIT_SIZE + CELL_UNIT_SIZE / 2;
            const cellValueForComparison = String(cellValue).trim().toLowerCase();

            if (cellValueForComparison === "o") {
                const markO = document.createElementNS(SVG_NS, "circle");
                markO.setAttribute("cx", cx.toString());
                markO.setAttribute("cy", cy.toString());
                markO.setAttribute("r", O_MARK_RADIUS.toString());
                markO.setAttribute("stroke", MARK_COLOR);
                markO.setAttribute("stroke-width", MARK_THICKNESS.toString());
                markO.setAttribute("fill", "none");
                markO.classList.add("game-mark");
                currentBoardSvgElement.appendChild(markO);
            } else if (cellValueForComparison === "x") {
                const line1 = document.createElementNS(SVG_NS, "line");
                line1.setAttribute("x1", (cx - X_MARK_ARM_LENGTH).toString());
                line1.setAttribute("y1", (cy - X_MARK_ARM_LENGTH).toString());
                line1.setAttribute("x2", (cx + X_MARK_ARM_LENGTH).toString());
                line1.setAttribute("y2", (cy + X_MARK_ARM_LENGTH).toString());
                line1.setAttribute("stroke", MARK_COLOR);
                line1.setAttribute("stroke-width", MARK_THICKNESS.toString());
                line1.classList.add("game-mark");
                currentBoardSvgElement.appendChild(line1);

                const line2 = document.createElementNS(SVG_NS, "line");
                line2.setAttribute("x1", (cx + X_MARK_ARM_LENGTH).toString());
                line2.setAttribute("y1", (cy - X_MARK_ARM_LENGTH).toString());
                line2.setAttribute("x2", (cx - X_MARK_ARM_LENGTH).toString());
                line2.setAttribute("y2", (cy + X_MARK_ARM_LENGTH).toString());
                line2.setAttribute("stroke", MARK_COLOR);
                line2.setAttribute("stroke-width", MARK_THICKNESS.toString());
                line2.classList.add("game-mark");
                currentBoardSvgElement.appendChild(line2);
            }
        }

        currentStatusTextElement.innerHTML = '';
        currentWinnerTextElement.innerHTML = '';
        const spanStyle = 'font-weight: bold; color: ' + MARK_COLOR + ';';

        if (is_terminal) {
            currentStatusTextElement.textContent = '';
            if (winner !== null && winner !== undefined) {
                if (String(winner).toLowerCase() === 'draw') {
                    currentWinnerTextElement.textContent = "It's a Draw!";
                } else {
                    let winnerSymbolDisplay;
                    if (String(winner).toLowerCase() === "o") { winnerSymbolDisplay = PLAYER_SYMBOLS[0]; }
                    else if (String(winner).toLowerCase() === "x") { winnerSymbolDisplay = PLAYER_SYMBOLS[1]; }
                    if (winnerSymbolDisplay) {
                        currentWinnerTextElement.innerHTML = 'Player <span style="' + spanStyle + '">' + winnerSymbolDisplay + '</span> Wins!';
                    } else {
                        currentWinnerTextElement.textContent = `Winner: ${String(winner).toUpperCase()}`;
                    }
                }
            } else { currentWinnerTextElement.textContent = "Game Over!"; }
        } else {
            let playerSymbolToDisplay;
            if (String(current_player).toLowerCase() === "o") { playerSymbolToDisplay = PLAYER_SYMBOLS[0]; }
            else if (String(current_player).toLowerCase() === "x") { playerSymbolToDisplay = PLAYER_SYMBOLS[1]; }
            if (playerSymbolToDisplay) {
                currentStatusTextElement.innerHTML = 'Current Player: <span style="' + spanStyle + '">' + playerSymbolToDisplay + '</span>';
            } else {
                currentStatusTextElement.textContent = "Waiting for player...";
            }
        }
    }

    if (!_ensureRendererElements(parent, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS)) {
        if (parent && typeof parent.innerHTML !== 'undefined') {
            parent.innerHTML = "<p style='color:red; font-family: sans-serif;'>Critical Error: Renderer element setup failed.</p>";
        }
        console.error("Renderer element setup failed and parent could not be updated.");
        return;
    }

    if (!environment || !environment.steps || !environment.steps[step]) {
        _renderBoardDisplay_svg(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if(currentStatusTextElement) currentStatusTextElement.textContent = "Initializing environment...";
        return;
    }

    const currentStepAgents = environment.steps[step];
    if (!currentStepAgents || !Array.isArray(currentStepAgents) || currentStepAgents.length === 0) {
        _renderBoardDisplay_svg(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if(currentStatusTextElement) currentStatusTextElement.textContent = "Waiting for agent data...";
        return;
    }

    const environmentAgentIndex = currentStepAgents.length - 1;
    const environmentAgent = currentStepAgents[environmentAgentIndex];

    if (!environmentAgent || typeof environmentAgent.observation === 'undefined') {
        _renderBoardDisplay_svg(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if(currentStatusTextElement) currentStatusTextElement.textContent = "Waiting for observation data...";
        return;
    }

    const observationForBoardState = environmentAgent.observation;
    let gameSpecificState = null;

    if (observationForBoardState) {
        let boardArray = null;
        let currentPlayerForState = null;

        if (typeof observationForBoardState.observationString === 'string' && observationForBoardState.observationString.trim().startsWith('{')) {
            try {
                const parsedState = JSON.parse(observationForBoardState.observationString);
                if (parsedState && Array.isArray(parsedState.board)) {
                    boardArray = parsedState.board;
                }
                if (parsedState && typeof parsedState.current_player === 'string') {
                    currentPlayerForState = parsedState.current_player;
                }
            } catch (e) {
                _showMessage("Error parsing game state from observation string.", 'error');
                console.error("Failed to parse observation string JSON:", e);
            }
        }

        if (!boardArray) {
            boardArray = [];
            for (let i = 0; i < DEFAULT_NUM_ROWS * DEFAULT_NUM_COLS; i++) {
                boardArray.push(null);
            }
        }

        if (!currentPlayerForState) {
             if (observationForBoardState.currentPlayer === 0) { currentPlayerForState = 'x'; }
             else if (observationForBoardState.currentPlayer === 1) { currentPlayerForState = 'o'; }
        }

        const isTerminal = !!observationForBoardState.is_terminal;
        let winnerForState = null;

        if (isTerminal) {
            const finalRewards = window.kaggle && window.kaggle.rewards;
            if (finalRewards && Array.isArray(finalRewards)) {
                if (finalRewards[0] === 1.0) winnerForState = 'x';
                else if (finalRewards[1] === 1.0) winnerForState = 'o';
                else if (finalRewards[0] === 0.0 && finalRewards[1] === 0.0) winnerForState = 'draw';
            }
            if (observationForBoardState.currentPlayer === -2 && winnerForState === null) {
                winnerForState = 'draw';
            }
        }

        gameSpecificState = {
            board: boardArray,
            current_player: currentPlayerForState,
            is_terminal: isTerminal,
            winner: winnerForState
        };
    }

    if (!gameSpecificState) {
        _renderBoardDisplay_svg(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if(currentStatusTextElement) currentStatusTextElement.textContent = "Error processing game state.";
        _showMessage("Error: Game state could not be parsed correctly.", 'error');
        console.error("Could not determine gameSpecificState from:", observationForBoardState);
        return;
    }

    _renderBoardDisplay_svg(gameSpecificState, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
}