function renderer(options) {
    const { environment, step, parent, interactive, isInteractive } = options;

    const DEFAULT_NUM_ROWS = 6;
    const DEFAULT_NUM_COLS = 7;
    const PLAYER_SYMBOLS = ['O', 'X']; // O: Player 0 (Yellow), X: Player 1 (Red)
    const PLAYER_COLORS = ['#facc15', '#ef4444']; // Yellow for 'O', Red for 'X'
    const EMPTY_CELL_COLOR = '#e5e7eb'; 
    const BOARD_COLOR = '#3b82f6';      

    const SVG_NS = "http://www.w3.org/2000/svg";
    const CELL_UNIT_SIZE = 100; 
    const CIRCLE_RADIUS = CELL_UNIT_SIZE * 0.42; 
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
            currentMessageBoxElement.style.top = '10px';
            currentMessageBoxElement.style.left = '50%';
            currentMessageBoxElement.style.transform = 'translateX(-50%)';
            currentMessageBoxElement.style.padding = '0.75rem 1rem';
            currentMessageBoxElement.style.borderRadius = '0.375rem';
            currentMessageBoxElement.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            currentMessageBoxElement.style.zIndex = '1000';
            currentMessageBoxElement.style.opacity = '0';
            currentMessageBoxElement.style.transition = 'opacity 0.3s ease-in-out, background-color 0.3s';
            currentMessageBoxElement.style.fontSize = '0.875rem';
            currentMessageBoxElement.style.fontFamily = "'Inter', sans-serif";
            document.body.appendChild(currentMessageBoxElement);
        }
        currentMessageBoxElement.textContent = message;
        currentMessageBoxElement.style.backgroundColor = type === 'error' ? '#ef4444' : '#10b981';
        currentMessageBoxElement.style.color = 'white';
        currentMessageBoxElement.style.opacity = '1';
        setTimeout(() => { if (currentMessageBoxElement) currentMessageBoxElement.style.opacity = '0'; }, duration);
    }

    function _ensureRendererElements(parentElementToClear, rows, cols) {
        if (!parentElementToClear) return false;
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
        currentTitleElement.textContent = 'Connect Four';
        currentTitleElement.style.fontSize = '1.875rem';
        currentTitleElement.style.fontWeight = 'bold';
        currentTitleElement.style.marginBottom = '1rem';
        currentTitleElement.style.textAlign = 'center';
        currentTitleElement.style.color = '#2563eb';
        currentRendererContainer.appendChild(currentTitleElement);

        currentBoardSvgElement = document.createElementNS(SVG_NS, "svg");
        currentBoardSvgElement.setAttribute("viewBox", `0 0 ${SVG_VIEWBOX_WIDTH} ${SVG_VIEWBOX_HEIGHT}`);
        currentBoardSvgElement.setAttribute("preserveAspectRatio", "xMidYMid meet");
        currentBoardSvgElement.style.width = "auto"; 
        currentBoardSvgElement.style.maxWidth = "500px"; 
        currentBoardSvgElement.style.maxHeight = `calc(100vh - 200px)`; 
        currentBoardSvgElement.style.aspectRatio = `${cols} / ${rows}`;
        currentBoardSvgElement.style.display = "block"; 
        currentBoardSvgElement.style.margin = "0 auto 20px auto"; 

        const boardBgRect = document.createElementNS(SVG_NS, "rect");
        boardBgRect.setAttribute("x", "0");
        boardBgRect.setAttribute("y", "0");
        boardBgRect.setAttribute("width", SVG_VIEWBOX_WIDTH.toString());
        boardBgRect.setAttribute("height", SVG_VIEWBOX_HEIGHT.toString());
        boardBgRect.setAttribute("fill", BOARD_COLOR);
        boardBgRect.setAttribute("rx", (CELL_UNIT_SIZE * 0.1).toString()); 
        currentBoardSvgElement.appendChild(boardBgRect);

        // SVG Circles are created with (0,0) being top-left visual circle
        for (let r_visual = 0; r_visual < rows; r_visual++) {
            for (let c_visual = 0; c_visual < cols; c_visual++) {
                const circle = document.createElementNS(SVG_NS, "circle");
                const cx = c_visual * CELL_UNIT_SIZE + CELL_UNIT_SIZE / 2;
                const cy = r_visual * CELL_UNIT_SIZE + CELL_UNIT_SIZE / 2;
                circle.setAttribute("id", `cell-${r_visual}-${c_visual}`);
                circle.setAttribute("cx", cx.toString());
                circle.setAttribute("cy", cy.toString());
                circle.setAttribute("r", CIRCLE_RADIUS.toString());
                circle.setAttribute("fill", EMPTY_CELL_COLOR);
                currentBoardSvgElement.appendChild(circle);
            }
        }
        currentRendererContainer.appendChild(currentBoardSvgElement);

        const statusContainer = document.createElement('div');
        statusContainer.style.padding = '10px 15px';
        statusContainer.style.backgroundColor = 'white';
        statusContainer.style.borderRadius = '8px';
        statusContainer.style.boxShadow = '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)';
        statusContainer.style.textAlign = 'center';
        statusContainer.style.width = 'auto';
        statusContainer.style.minWidth = '200px'; 
        statusContainer.style.maxWidth = '90vw';
        currentRendererContainer.appendChild(statusContainer);

        currentStatusTextElement = document.createElement('p');
        currentStatusTextElement.style.fontSize = '1.1rem';
        currentStatusTextElement.style.fontWeight = '600';
        currentStatusTextElement.style.margin = '0 0 5px 0';
        statusContainer.appendChild(currentStatusTextElement);
        
        currentWinnerTextElement = document.createElement('p');
        currentWinnerTextElement.style.fontSize = '1.25rem';
        currentWinnerTextElement.style.fontWeight = '700';
        currentWinnerTextElement.style.margin = '5px 0 0 0';
        statusContainer.appendChild(currentWinnerTextElement);
        
        parentElementToClear.appendChild(currentRendererContainer);
        
        if (typeof document !== 'undefined' && !document.body.hasAttribute('data-renderer-initialized')) {
             document.body.setAttribute('data-renderer-initialized', 'true');
        }
        return true;
    }

    function _renderBoardDisplay_svg(gameStateToDisplay, displayRows, displayCols) {
        if (!currentBoardSvgElement || !currentStatusTextElement || !currentWinnerTextElement) return;

        if (!gameStateToDisplay || typeof gameStateToDisplay.board !== 'object' || !Array.isArray(gameStateToDisplay.board) || gameStateToDisplay.board.length === 0) {
            currentStatusTextElement.textContent = "Waiting for game data...";
            currentWinnerTextElement.textContent = "";
            for (let r_visual = 0; r_visual < displayRows; r_visual++) {
                for (let c_visual = 0; c_visual < displayCols; c_visual++) {
                    const circleElement = currentBoardSvgElement.querySelector(`#cell-${r_visual}-${c_visual}`);
                    if (circleElement) {
                        circleElement.setAttribute("fill", EMPTY_CELL_COLOR);
                    }
                }
            }
            return;
        }

        const { board, current_player, is_terminal, winner } = gameStateToDisplay;

        for (let r_data = 0; r_data < displayRows; r_data++) {
            const dataRow = board[r_data]; 
            if (!dataRow || !Array.isArray(dataRow) || dataRow.length !== displayCols) {
                // Error handling for malformed row
                for (let c_fill = 0; c_fill < displayCols; c_fill++) {
                    // Determine visual row for error display. If r_data=0 is top data,
                    // and we want to flip, then this error is for visual row (displayRows-1)-0.
                    const visual_row_for_error = (displayRows - 1) - r_data;
                    const circleElement = currentBoardSvgElement.querySelector(`#cell-${visual_row_for_error}-${c_fill}`);
                    if (circleElement) circleElement.setAttribute("fill", '#FF00FF'); // Magenta for error
                }
                continue;
            }

            const visual_svg_row_index = (displayRows - 1) - r_data;

            for (let c_data = 0; c_data < displayCols; c_data++) { // c_data iterates through columns of `board[r_data]`
                const originalCellValue = dataRow[c_data];
                const cellValueForComparison = String(originalCellValue).trim().toLowerCase();
                
                // The column index for SVG is the same as c_data
                const visual_svg_col_index = c_data;
                const circleElement = currentBoardSvgElement.querySelector(`#cell-${visual_svg_row_index}-${visual_svg_col_index}`);
                
                if (!circleElement) continue;
                
                let fillColor = EMPTY_CELL_COLOR;
                if (cellValueForComparison === "o") { 
                    fillColor = PLAYER_COLORS[0]; // Yellow
                } else if (cellValueForComparison === "x") { 
                    fillColor = PLAYER_COLORS[1]; // Red
                }
                circleElement.setAttribute("fill", fillColor);
            }
        }

        currentStatusTextElement.innerHTML = '';
        currentWinnerTextElement.innerHTML = '';
        if (is_terminal) {
            currentStatusTextElement.textContent = "Game Over!";
            if (winner !== null && winner !== undefined) {
                if (String(winner).toLowerCase() === 'draw') {
                    currentWinnerTextElement.textContent = "It's a Draw!";
                } else {
                    let winnerSymbolDisplay, winnerColorDisplay;
                    if (String(winner).toLowerCase() === "o") {
                        winnerSymbolDisplay = PLAYER_SYMBOLS[0]; 
                        winnerColorDisplay = PLAYER_COLORS[0];   
                    } else if (String(winner).toLowerCase() === "x") {
                        winnerSymbolDisplay = PLAYER_SYMBOLS[1]; 
                        winnerColorDisplay = PLAYER_COLORS[1];   
                    }
                    if (winnerSymbolDisplay) {
                         currentWinnerTextElement.innerHTML = `Player <span style="color: ${winnerColorDisplay}; font-weight: bold;">${winnerSymbolDisplay}</span> Wins!`;
                    } else {
                        currentWinnerTextElement.textContent = `Winner: ${String(winner).toUpperCase()}`; 
                    }
                }
            } else { currentWinnerTextElement.textContent = "Game ended."; }
        } else { 
            let playerSymbolToDisplay, playerColorToDisplay;
            if (String(current_player).toLowerCase() === "o") {
                playerSymbolToDisplay = PLAYER_SYMBOLS[0]; 
                playerColorToDisplay = PLAYER_COLORS[0];   
            } else if (String(current_player).toLowerCase() === "x") {
                playerSymbolToDisplay = PLAYER_SYMBOLS[1]; 
                playerColorToDisplay = PLAYER_COLORS[1];   
            }
            if (playerSymbolToDisplay) {
                currentStatusTextElement.innerHTML = `Current Player: <span style="color: ${playerColorToDisplay}; font-weight: bold;">${playerSymbolToDisplay}</span>`;
            } else {
                currentStatusTextElement.textContent = "Waiting for player...";
            }
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
    
    const gameMasterAgentIndex = currentStepAgents.length - 1;
    const gameMasterAgent = currentStepAgents[gameMasterAgentIndex];

    if (!gameMasterAgent || typeof gameMasterAgent.observation === 'undefined') {
        _renderBoardDisplay_svg(null, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
        if(currentStatusTextElement) currentStatusTextElement.textContent = "Waiting for observation data...";
        return;
    }
    const observationForRenderer = gameMasterAgent.observation;

    let gameSpecificState = null;

    if (observationForRenderer && typeof observationForRenderer.observationString === 'string' && observationForRenderer.observationString.trim() !== '') {
        try {
            gameSpecificState = JSON.parse(observationForRenderer.observationString);
        } catch (e) {
            _showMessage("Error: Corrupted game state (obs_string).", 'error');
        }
    }
    
    if (!gameSpecificState && observationForRenderer && typeof observationForRenderer.json === 'string' && observationForRenderer.json.trim() !== '') {
        try {
            gameSpecificState = JSON.parse(observationForRenderer.json);
        } catch (e) {
            _showMessage("Error: Corrupted game state (json).", 'error');
        }
    }
    
    _renderBoardDisplay_svg(gameSpecificState, DEFAULT_NUM_ROWS, DEFAULT_NUM_COLS);
}