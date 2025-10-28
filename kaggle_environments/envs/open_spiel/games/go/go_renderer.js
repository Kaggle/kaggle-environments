// Go Board Renderer for OpenSpiel
function renderer(options) {
    const { environment, step, parent, interactive, isInteractive, maxBoardSize = 800 } = options;

    // --- Constants ---
    const DEFAULT_BOARD_SIZE = 19;
    const STONE_COLORS = {
        'B': '#2d3748', // Black stone
        'W': '#f7fafc', // White stone with slight gray tint
        '.': 'transparent' // Empty intersection
    };
    const BOARD_COLOR = '#dcb871'; // Traditional Go board wood color
    const LINE_COLOR = '#8b4513'; // Dark brown for grid lines
    const STAR_POINT_COLOR = '#654321'; // Darker brown for star points
    const LABEL_COLOR = '#2d3748';

    const SVG_NS = "http://www.w3.org/2000/svg";
    
    // Dynamic sizing based on board size
    function getBoardConfig(boardSize) {
        let intersectionSize, margin, fontSize;
        
        if (boardSize <= 9) {
            intersectionSize = 35;
            margin = 45;
            fontSize = 14;
        } else if (boardSize <= 13) {
            intersectionSize = 30;
            margin = 40;
            fontSize = 13;
        } else {
            intersectionSize = 25;
            margin = 35;
            fontSize = 12;
        }
        
        return { intersectionSize, margin, fontSize };
    }
    
    // Go column labels (A-T, omitting I)
    const COLUMN_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'];

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

    function _getStarPoints(boardSize) {
        // Star points for different board sizes
        const starPointsMap = {
            9: [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]],
            13: [[3, 3], [3, 9], [6, 6], [9, 3], [9, 9]],
            19: [[3, 3], [3, 9], [3, 15], [9, 3], [9, 9], [9, 15], [15, 3], [15, 9], [15, 15]]
        };
        return starPointsMap[boardSize] || [];
    }

    function _ensureRendererElements(parentElementToClear, boardSize) {
        if (!parentElementToClear) return false;
        parentElementToClear.innerHTML = '';

        const config = getBoardConfig(boardSize);
        const { intersectionSize, margin, fontSize } = config;

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
        currentTitleElement.textContent = `Go (${boardSize}×${boardSize})`;
        currentTitleElement.style.fontSize = '1.875rem';
        currentTitleElement.style.fontWeight = 'bold';
        currentTitleElement.style.marginBottom = '1rem';
        currentTitleElement.style.textAlign = 'center';
        currentTitleElement.style.color = '#2563eb';
        currentRendererContainer.appendChild(currentTitleElement);

        // Calculate SVG dimensions
        const boardPixelSize = (boardSize - 1) * intersectionSize;
        const svgWidth = boardPixelSize + (2 * margin);
        const svgHeight = boardPixelSize + (2 * margin);

        currentBoardSvgElement = document.createElementNS(SVG_NS, "svg");
        currentBoardSvgElement.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);
        currentBoardSvgElement.setAttribute("preserveAspectRatio", "xMidYMid meet");
        currentBoardSvgElement.style.width = "auto";
        currentBoardSvgElement.style.maxWidth = `${maxBoardSize}px`;
        currentBoardSvgElement.style.maxHeight = `${maxBoardSize}px`;
        currentBoardSvgElement.style.aspectRatio = "1 / 1";
        currentBoardSvgElement.style.display = "block";
        currentBoardSvgElement.style.margin = "0 auto 20px auto";

        // Board background
        const boardBgRect = document.createElementNS(SVG_NS, "rect");
        boardBgRect.setAttribute("x", "0");
        boardBgRect.setAttribute("y", "0");
        boardBgRect.setAttribute("width", svgWidth.toString());
        boardBgRect.setAttribute("height", svgHeight.toString());
        boardBgRect.setAttribute("fill", BOARD_COLOR);
        boardBgRect.setAttribute("rx", "8");
        currentBoardSvgElement.appendChild(boardBgRect);

        // Grid lines
        for (let i = 0; i < boardSize; i++) {
            const x = margin + i * intersectionSize;
            const y = margin + i * intersectionSize;
            
            // Vertical lines
            const vLine = document.createElementNS(SVG_NS, "line");
            vLine.setAttribute("x1", x.toString());
            vLine.setAttribute("y1", margin.toString());
            vLine.setAttribute("x2", x.toString());
            vLine.setAttribute("y2", (margin + boardPixelSize).toString());
            vLine.setAttribute("stroke", LINE_COLOR);
            vLine.setAttribute("stroke-width", "1");
            currentBoardSvgElement.appendChild(vLine);
            
            // Horizontal lines
            const hLine = document.createElementNS(SVG_NS, "line");
            hLine.setAttribute("x1", margin.toString());
            hLine.setAttribute("y1", y.toString());
            hLine.setAttribute("x2", (margin + boardPixelSize).toString());
            hLine.setAttribute("y2", y.toString());
            hLine.setAttribute("stroke", LINE_COLOR);
            hLine.setAttribute("stroke-width", "1");
            currentBoardSvgElement.appendChild(hLine);
        }

        // Star points
        const starPoints = _getStarPoints(boardSize);
        starPoints.forEach(([row, col]) => {
            const x = margin + col * intersectionSize;
            const y = margin + row * intersectionSize;
            const starPoint = document.createElementNS(SVG_NS, "circle");
            starPoint.setAttribute("cx", x.toString());
            starPoint.setAttribute("cy", y.toString());
            starPoint.setAttribute("r", "3");
            starPoint.setAttribute("fill", STAR_POINT_COLOR);
            currentBoardSvgElement.appendChild(starPoint);
        });

        // Coordinate labels
        const labelOffset = Math.max(15, margin * 0.6);
        
        for (let i = 0; i < boardSize; i++) {
            // Column labels (top and bottom)
            if (i < COLUMN_LABELS.length) {
                const x = margin + i * intersectionSize;
                
                // Top labels
                const topLabel = document.createElementNS(SVG_NS, "text");
                topLabel.setAttribute("x", x.toString());
                topLabel.setAttribute("y", labelOffset.toString());
                topLabel.setAttribute("text-anchor", "middle");
                topLabel.setAttribute("dominant-baseline", "middle");
                topLabel.setAttribute("font-family", "Arial, sans-serif");
                topLabel.setAttribute("font-size", fontSize.toString());
                topLabel.setAttribute("fill", LABEL_COLOR);
                topLabel.textContent = COLUMN_LABELS[i];
                currentBoardSvgElement.appendChild(topLabel);
                
                // Bottom labels
                const bottomLabel = document.createElementNS(SVG_NS, "text");
                bottomLabel.setAttribute("x", x.toString());
                bottomLabel.setAttribute("y", (svgHeight - labelOffset + 5).toString());
                bottomLabel.setAttribute("text-anchor", "middle");
                bottomLabel.setAttribute("dominant-baseline", "middle");
                bottomLabel.setAttribute("font-family", "Arial, sans-serif");
                bottomLabel.setAttribute("font-size", fontSize.toString());
                bottomLabel.setAttribute("fill", LABEL_COLOR);
                bottomLabel.textContent = COLUMN_LABELS[i];
                currentBoardSvgElement.appendChild(bottomLabel);
            }

            // Row labels (left and right) - Go rows are numbered from bottom to top
            const rowNumber = boardSize - i;
            const y = margin + i * intersectionSize;
            
            // Left labels
            const leftLabel = document.createElementNS(SVG_NS, "text");
            leftLabel.setAttribute("x", labelOffset.toString());
            leftLabel.setAttribute("y", y.toString());
            leftLabel.setAttribute("text-anchor", "middle");
            leftLabel.setAttribute("dominant-baseline", "middle");
            leftLabel.setAttribute("font-family", "Arial, sans-serif");
            leftLabel.setAttribute("font-size", fontSize.toString());
            leftLabel.setAttribute("fill", LABEL_COLOR);
            leftLabel.textContent = rowNumber.toString();
            currentBoardSvgElement.appendChild(leftLabel);
            
            // Right labels
            const rightLabel = document.createElementNS(SVG_NS, "text");
            rightLabel.setAttribute("x", (svgWidth - labelOffset).toString());
            rightLabel.setAttribute("y", y.toString());
            rightLabel.setAttribute("text-anchor", "middle");
            rightLabel.setAttribute("dominant-baseline", "middle");
            rightLabel.setAttribute("font-family", "Arial, sans-serif");
            rightLabel.setAttribute("font-size", fontSize.toString());
            rightLabel.setAttribute("fill", LABEL_COLOR);
            rightLabel.textContent = rowNumber.toString();
            currentBoardSvgElement.appendChild(rightLabel);
        }

        // Create intersection circles for stones
        const stoneRadius = Math.max(8, intersectionSize * 0.4);
        
        for (let row = 0; row < boardSize; row++) {
            for (let col = 0; col < boardSize; col++) {
                const x = margin + col * intersectionSize;
                const y = margin + row * intersectionSize;
                const stone = document.createElementNS(SVG_NS, "circle");
                stone.setAttribute("id", `stone-${row}-${col}`);
                stone.setAttribute("cx", x.toString());
                stone.setAttribute("cy", y.toString());
                stone.setAttribute("r", stoneRadius.toString());
                stone.setAttribute("fill", "transparent");
                stone.setAttribute("stroke", "none");
                currentBoardSvgElement.appendChild(stone);
                
                // Create a smaller circle for indicating the most recent move
                const recentMoveRadius = Math.max(3, stoneRadius * 0.55);
                const recentMoveIndicator = document.createElementNS(SVG_NS, "circle");
                recentMoveIndicator.setAttribute("id", `recent-move-${row}-${col}`);
                recentMoveIndicator.setAttribute("cx", x.toString());
                recentMoveIndicator.setAttribute("cy", y.toString());
                recentMoveIndicator.setAttribute("r", recentMoveRadius.toString());
                recentMoveIndicator.setAttribute("fill", "transparent");
                recentMoveIndicator.setAttribute("stroke", "none");
                currentBoardSvgElement.appendChild(recentMoveIndicator);
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
            _showMessage("Go Renderer initialized.", "info", 1500);
            document.body.setAttribute('data-renderer-initialized', 'true');
        }
        return true;
    }

    function _renderBoardDisplay_svg(gameStateToDisplay, boardSize) {
        if (!currentBoardSvgElement || !currentStatusTextElement || !currentWinnerTextElement) return;

        // Clear all stones and recent move indicators first
        for (let row = 0; row < boardSize; row++) {
            for (let col = 0; col < boardSize; col++) {
                const stoneElement = currentBoardSvgElement.querySelector(`#stone-${row}-${col}`);
                if (stoneElement) {
                    stoneElement.setAttribute("fill", "transparent");
                    stoneElement.setAttribute("stroke", "none");
                }
                
                const recentMoveElement = currentBoardSvgElement.querySelector(`#recent-move-${row}-${col}`);
                if (recentMoveElement) {
                    recentMoveElement.setAttribute("fill", "transparent");
                    recentMoveElement.setAttribute("stroke", "none");
                }
            }
        }

        if (!gameStateToDisplay || !gameStateToDisplay.board_grid) {
            currentStatusTextElement.textContent = "Waiting for game data...";
            currentWinnerTextElement.textContent = "";
            return;
        }

        const { board_grid, current_player_to_move, move_number, komi, previous_move_a1 } = gameStateToDisplay;

        // Render stones on the board
        // board_grid[0] is the top row (row 9 in a 9x9 board), board_grid[8] is bottom row (row 1)
        for (let gridRow = 0; gridRow < board_grid.length && gridRow < boardSize; gridRow++) {
            const rowData = board_grid[gridRow];
            if (!Array.isArray(rowData)) continue;

            for (let gridCol = 0; gridCol < rowData.length && gridCol < boardSize; gridCol++) {
                const intersection = rowData[gridCol];
                if (!intersection || typeof intersection !== 'object') continue;

                // Extract the stone state from the intersection dictionary
                const coordinate = Object.keys(intersection)[0];
                const stoneState = intersection[coordinate];

                if (stoneState && (stoneState === 'B' || stoneState === 'W')) {
                    const stoneElement = currentBoardSvgElement.querySelector(`#stone-${gridRow}-${gridCol}`);
                    if (stoneElement) {
                        stoneElement.setAttribute("fill", STONE_COLORS[stoneState]);
                        stoneElement.setAttribute("stroke", stoneState === 'W' ? '#666' : 'none');
                        stoneElement.setAttribute("stroke-width", stoneState === 'W' ? "1" : "0");
                    }
                }
            }
        }

        // Highlight the most recent move if available
        if (previous_move_a1) {
            // Parse the coordinate (e.g., "F4" -> column F, row 4)
            const colLetter = previous_move_a1[0];
            const rowNumber = parseInt(previous_move_a1.slice(1));
            
            // Convert to array indices
            const COLUMN_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'];
            const colIndex = COLUMN_LABELS.indexOf(colLetter);
            const rowIndex = boardSize - rowNumber; // Convert Go row numbering to array index
            
            if (rowIndex >= 0 && rowIndex < boardSize && colIndex >= 0 && colIndex < boardSize) {
                const recentMoveElement = currentBoardSvgElement.querySelector(`#recent-move-${rowIndex}-${colIndex}`);
                const stoneElement = currentBoardSvgElement.querySelector(`#stone-${rowIndex}-${colIndex}`);
                
                if (recentMoveElement && stoneElement) {
                    // Get the color of the stone at this position to determine the indicator color
                    const stoneFill = stoneElement.getAttribute("fill");
                    
                    if (stoneFill === STONE_COLORS['B']) {
                        // Black stone - use white circle outline
                        recentMoveElement.setAttribute("fill", "transparent");
                        recentMoveElement.setAttribute("stroke", STONE_COLORS['W']);
                        recentMoveElement.setAttribute("stroke-width", "1.25");
                    } else if (stoneFill === STONE_COLORS['W']) {
                        // White stone - use black circle outline
                        recentMoveElement.setAttribute("fill", "transparent");
                        recentMoveElement.setAttribute("stroke", STONE_COLORS['B']);
                        recentMoveElement.setAttribute("stroke-width", "1.25");
                    }
                }
            }
        }

        // Update status display
        const playerColor = current_player_to_move === 'B' ? '#2d3748' : '#f7fafc';
        const playerName = current_player_to_move === 'B' ? 'Black' : 'White';
        
        currentStatusTextElement.innerHTML = `Move ${move_number || 1}: <span style="color: ${playerColor}; font-weight: bold; ${current_player_to_move === 'W' ? 'text-shadow: 1px 1px 2px rgba(0,0,0,0.3);' : ''}">${playerName}</span> to play`;
        
        if (previous_move_a1) {
            currentWinnerTextElement.textContent = `Last move: ${previous_move_a1}${komi ? ` • Komi: ${komi}` : ''}`;
        } else {
            currentWinnerTextElement.textContent = komi ? `Komi: ${komi}` : '';
        }
    }

    // --- Main execution logic ---
    let boardSize = DEFAULT_BOARD_SIZE;
    
    // Try to extract board size from game state
    if (environment && environment.steps && environment.steps[step]) {
        const currentStepAgents = environment.steps[step];
        if (Array.isArray(currentStepAgents) && currentStepAgents.length > 0) {
            const gameMasterAgent = currentStepAgents[currentStepAgents.length - 1];
            if (gameMasterAgent && gameMasterAgent.observation) {
                let gameState = null;
                
                // Try to parse game state from observation
                if (gameMasterAgent.observation.observation_string) {
                    try {
                        gameState = JSON.parse(gameMasterAgent.observation.observation_string);
                    } catch (e) {}
                }
                
                if (!gameState && gameMasterAgent.observation.json) {
                    try {
                        gameState = JSON.parse(gameMasterAgent.observation.json);
                    } catch (e) {}
                }
                
                if (gameState && gameState.board_size) {
                    boardSize = gameState.board_size;
                }
            }
        }
    }

    if (!_ensureRendererElements(parent, boardSize)) {
        if (parent && typeof parent.innerHTML !== 'undefined') {
            parent.innerHTML = "<p style='color:red; font-family: sans-serif;'>Critical Error: Renderer element setup failed.</p>";
        }
        return;
    }
    
    if (!environment || !environment.steps || !environment.steps[step]) {
        _renderBoardDisplay_svg(null, boardSize);
        if(currentStatusTextElement) currentStatusTextElement.textContent = "Initializing environment...";
        return;
    }

    const currentStepAgents = environment.steps[step];
    if (!currentStepAgents || !Array.isArray(currentStepAgents) || currentStepAgents.length === 0) {
        _renderBoardDisplay_svg(null, boardSize);
        if(currentStatusTextElement) currentStatusTextElement.textContent = "Waiting for agent data...";
        return;
    }
    
    const gameMasterAgentIndex = currentStepAgents.length - 1;
    const gameMasterAgent = currentStepAgents[gameMasterAgentIndex];

    if (!gameMasterAgent || typeof gameMasterAgent.observation === 'undefined') {
        _renderBoardDisplay_svg(null, boardSize);
        if(currentStatusTextElement) currentStatusTextElement.textContent = "Waiting for observation data...";
        return;
    }
    
    const observationForRenderer = gameMasterAgent.observation;
    let gameSpecificState = null;

    if (observationForRenderer && typeof observationForRenderer.observation_string === 'string' && observationForRenderer.observation_string.trim() !== '') {
        try {
            gameSpecificState = JSON.parse(observationForRenderer.observation_string);
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
    
    _renderBoardDisplay_svg(gameSpecificState, boardSize);
}