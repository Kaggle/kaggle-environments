import type { RendererOptions } from '@kaggle-environments/core';

interface VisualizerState {
  initialized: boolean;
  isPlaying: boolean;
  activeStep: number;
  hoveredCell: { subgrid: number; cell: number } | null;
  moves: ParsedMove[];
  animationStartTime: number;
  lastStepWithWinners: number;
  prevWinners: string[];
}

interface ParsedMove {
  index: number;
  stepIndex: number;
  player: 'x' | 'o';
  playerLabel: string;
  text: string;
  timestamp: string;
  subgridIdx: number;
  cellIdx: number | null;
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent, agents, setStep, setPlaying, registerPlaybackHandlers } = options;
  const steps = replay.steps as any[];

  if (!steps || steps.length === 0) return;

  // 1. Initialize or retrieve state on parent element
  let stateObj: VisualizerState = (parent as any).__visualizer_state__;
  if (!stateObj) {
    stateObj = {
      initialized: false,
      isPlaying: false,
      activeStep: step,
      hoveredCell: null,
      moves: [],
      animationStartTime: 0,
      lastStepWithWinners: -1,
      prevWinners: Array(9).fill(''),
    };
    (parent as any).__visualizer_state__ = stateObj;
  }

  const player1Name = agents?.[0]?.name || 'Player 1';
  const player2Name = agents?.[1]?.name || 'Player 2';

  // 2. Build parsed moves history once
  if (stateObj.moves.length === 0 && steps.length > 2) {
    const parsedMoves: ParsedMove[] = [];
    let moveCounter = 1;
    for (let t = 2; t < steps.length; t++) {
      const act0 = steps[t]?.[0]?.action;
      const act1 = steps[t]?.[1]?.action;
      let player: 'x' | 'o' | null = null;
      let actionString = '';
      let timeTaken = 0;

      if (act0 && (act0.actionString || act0.submission !== -1)) {
        player = 'x';
        actionString = act0.actionString || `Action ${act0.submission}`;
        timeTaken = steps[t]?.[0]?.info?.timeTaken || 0;
      } else if (act1 && (act1.actionString || act1.submission !== -1)) {
        player = 'o';
        actionString = act1.actionString || `Action ${act1.submission}`;
        timeTaken = steps[t]?.[1]?.info?.timeTaken || 0;
      }

      if (player && actionString) {
        let descriptiveText = '';
        const playerLabel = player === 'x' ? player1Name : player2Name;
        let subgridIdx = -1;
        let cellIdx: number | null = null;

        if (actionString.startsWith('Choose local board')) {
          const parts = actionString.split(' ');
          subgridIdx = parseInt(parts[parts.length - 1]);
          const row = Math.floor(subgridIdx / 3);
          const col = subgridIdx % 3;
          descriptiveText = `${playerLabel} chose Sub-grid [Row ${row}, Col ${col}]`;
        } else {
          const match = actionString.match(/Local board (\d+): [xo]\((\d+),(\d+)\)/);
          if (match) {
            subgridIdx = parseInt(match[1]);
            const cellRow = parseInt(match[2]);
            const cellCol = parseInt(match[3]);
            cellIdx = cellRow * 3 + cellCol;
            const subRow = Math.floor(subgridIdx / 3);
            const subCol = subgridIdx % 3;
            descriptiveText = `${playerLabel} placed at [Row ${cellRow}, Col ${cellCol}] of Sub-grid [Row ${subRow}, Col ${subCol}]`;
          } else {
            descriptiveText = `${playerLabel}: ${actionString}`;
          }
        }

        parsedMoves.push({
          index: moveCounter++,
          stepIndex: t,
          player,
          playerLabel: player === 'x' ? `${player1Name} (X)` : `${player2Name} (O)`,
          text: descriptiveText,
          timestamp: `${timeTaken.toFixed(1)}s`,
          subgridIdx,
          cellIdx,
        });
      }
    }
    stateObj.moves = parsedMoves;
  }

  // 3. Build UI structure lazily (run once)
  const existingContainer = parent.querySelector('.renderer-container') as HTMLDivElement;
  const isOurContainer = existingContainer && existingContainer.querySelector('.utt-left-pane');
  if (!isOurContainer) {
    parent.innerHTML = `
      <div class="renderer-container">
        <!-- Left Pane: Main Board and Info -->
        <div class="utt-left-pane">
          <div class="utt-turn-plaque-container header">
            <div class="utt-turn-plaque p0 player-card">
              <span class="symbol">✕</span>
              <span class="name">${player1Name}</span>
              <span class="stats">Moves: 0 | Wins: 0</span>
            </div>
            <div class="utt-turn-plaque-divider">vs</div>
            <div class="utt-turn-plaque p1 player-card">
              <span class="symbol">◯</span>
              <span class="name">${player2Name}</span>
              <span class="stats">Moves: 0 | Wins: 0</span>
            </div>
          </div>
          
          <div class="board-wrap">
            <canvas></canvas>
          </div>
          
          <div class="utt-context-help-container">
            <div class="utt-context-help status-container"></div>
          </div>
        </div>
        
        <!-- Right Pane: Sidebar Tabbed Interface -->
        <div class="utt-right-pane">
          <!-- Tabs Header Selector -->
          <div class="utt-tabs-header">
            <button class="utt-tab-btn active" data-tab="match">Match View</button>
            <button class="utt-tab-btn" data-tab="history">Full Log</button>
          </div>

          <!-- Tab Content 1: Match View (Default) -->
          <div class="utt-tab-content active" id="tab-match">
            <!-- Controls Section -->
            <div class="utt-sidebar-section utt-controls-section">
              <div class="utt-odometer-container">
                <span class="utt-odometer">00 / 00</span>
              </div>
              <div class="utt-media-bar">
                <button class="utt-btn utt-btn-first" title="First Step"><i class="material-icons">first_page</i></button>
                <button class="utt-btn utt-btn-prev" title="Step Back"><i class="material-icons">chevron_left</i></button>
                <button class="utt-btn utt-btn-play" title="Play/Pause"><i class="material-icons">play_arrow</i></button>
                <button class="utt-btn utt-btn-next" title="Step Forward"><i class="material-icons">chevron_right</i></button>
                <button class="utt-btn utt-btn-last" title="Last Step"><i class="material-icons">last_page</i></button>
              </div>
              <div class="utt-slider-wrapper">
                <input type="range" min="0" max="${steps.length - 1}" value="0" class="utt-slider" />
              </div>
            </div>

            <!-- Mini 5-Move Live History Feed -->
            <div class="utt-sidebar-section utt-feed-section">
              <div class="utt-section-header">Live Move Feed</div>
              <div class="utt-mini-history-feed"></div>
            </div>
            
            <!-- Game Summary Section -->
            <div class="utt-sidebar-section utt-summary-section">
              <div class="utt-section-header">Game Summary</div>
              <div class="utt-summary-grid-container">
                <div class="utt-mini-grid"></div>
                <div class="utt-score-board">
                  <div class="utt-score-row p0">${player1Name}: 0 wins</div>
                  <div class="utt-score-row p1">${player2Name}: 0 wins</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Tab Content 2: Full Log (Scrollable) -->
          <div class="utt-tab-content" id="tab-history">
            <div class="utt-sidebar-section utt-log-section" style="flex: 1; min-height: 0; display: flex; flex-direction: column;">
              <div class="utt-section-header">Full Game History</div>
              <div class="utt-log-list" style="flex: 1; overflow-y: auto;"></div>
            </div>
          </div>
        </div>
      </div>
    `;

    // A. Bind controls event listeners
    const container = parent.querySelector('.renderer-container') as HTMLDivElement;
    const playBtn = container.querySelector('.utt-btn-play') as HTMLButtonElement;
    const prevBtn = container.querySelector('.utt-btn-prev') as HTMLButtonElement;
    const nextBtn = container.querySelector('.utt-btn-next') as HTMLButtonElement;
    const firstBtn = container.querySelector('.utt-btn-first') as HTMLButtonElement;
    const lastBtn = container.querySelector('.utt-btn-last') as HTMLButtonElement;
    const slider = container.querySelector('.utt-slider') as HTMLInputElement;

    playBtn.addEventListener('click', () => {
      setPlaying(!stateObj.isPlaying);
    });

    prevBtn.addEventListener('click', () => {
      setPlaying(false);
      setStep(Math.max(0, stateObj.activeStep - 1));
    });

    nextBtn.addEventListener('click', () => {
      setPlaying(false);
      setStep(Math.min(steps.length - 1, stateObj.activeStep + 1));
    });

    firstBtn.addEventListener('click', () => {
      setPlaying(false);
      setStep(0);
    });

    lastBtn.addEventListener('click', () => {
      setPlaying(false);
      setStep(steps.length - 1);
    });

    slider.addEventListener('input', (e) => {
      setPlaying(false);
      setStep(parseInt((e.target as HTMLInputElement).value));
    });

    // B. Render the static Log List items
    const logList = container.querySelector('.utt-log-list') as HTMLDivElement;
    logList.innerHTML = stateObj.moves
      .map(
        (m) => `
      <div class="utt-log-item" data-step="${m.stepIndex}">
        <span class="idx">${m.index}.</span>
        <span class="symbol ${m.player}">${m.player === 'x' ? '✕' : '◯'}</span>
        <span class="text">${m.text}</span>
        <span class="time">${m.timestamp}</span>
      </div>
    `
      )
      .join('');

    // Attach click listeners to log items
    logList.querySelectorAll('.utt-log-item').forEach((item) => {
      item.addEventListener('click', () => {
        setPlaying(false);
        setStep(parseInt(item.getAttribute('data-step') || '0'));
      });
    });

    // C. Setup Log Item Hover micro-view tooltip
    let tooltipEl: HTMLDivElement | null = null;
    logList.querySelectorAll('.utt-log-item').forEach((item) => {
      item.addEventListener('mouseenter', () => {
        const hoverStep = parseInt(item.getAttribute('data-step') || '0');

        // Create tooltip
        tooltipEl = document.createElement('div');
        tooltipEl.className = 'utt-log-tooltip';
        tooltipEl.innerHTML = `
          <span class="tooltip-title">Board State (Turn ${hoverStep})</span>
          <canvas width="120" height="120"></canvas>
        `;
        document.body.appendChild(tooltipEl);

        const canvas = tooltipEl.querySelector('canvas') as HTMLCanvasElement;
        drawMiniBoard(canvas, hoverStep, steps);

        // Position tooltip
        const rect = item.getBoundingClientRect();
        tooltipEl.style.top = `${rect.top - 130}px`;
        tooltipEl.style.left = `${rect.left + (rect.width - 130) / 2}px`;
      });

      item.addEventListener('mouseleave', () => {
        if (tooltipEl) {
          tooltipEl.remove();
          tooltipEl = null;
        }
      });
    });

    // D. Canvas mousemove listener for cell hover previews
    const wrapElement = container.querySelector('.board-wrap') as HTMLDivElement;
    wrapElement.addEventListener('mousemove', (e) => {
      const state = (parent as any).__visualizer_state__;
      const parsedObs = getObservationAtStep(state.activeStep, steps);
      const cv = wrapElement.querySelector('canvas');
      if (!cv) return;

      if (!parsedObs || parsedObs.is_terminal) {
        state.hoveredCell = null;
        cv.style.cursor = 'default';
        return;
      }

      const rect = cv.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const cssW = cv.width;
      const cssH = cv.height;
      const size = Math.min(cssW, cssH) - 40;
      const boardLeft = (cssW - size) / 2;
      const boardTop = Math.min(20, (cssH - size) / 2);

      if (mouseX >= boardLeft && mouseX <= boardLeft + size && mouseY >= boardTop && mouseY <= boardTop + size) {
        const relX = mouseX - boardLeft;
        const relY = mouseY - boardTop;
        const subgridSize = size / 3;
        const majorCol = Math.floor(relX / subgridSize);
        const majorRow = Math.floor(relY / subgridSize);
        const s = majorRow * 3 + majorCol;

        const subgridX = relX % subgridSize;
        const subgridY = relY % subgridSize;
        const cellSize = subgridSize / 3;
        const minorCol = Math.floor(subgridX / cellSize);
        const minorRow = Math.floor(subgridY / cellSize);
        const cellIdx = minorRow * 3 + minorCol;

        const activeSub = parsedObs.active_subgrid;
        const isPlayableSub =
          activeSub === null || activeSub === undefined ? parsedObs.subgrid_winners[s] === '' : s === activeSub;

        const isEmptyCell = parsedObs.board?.[s]?.[cellIdx] === '';

        if (isPlayableSub && isEmptyCell && parsedObs.subgrid_winners[s] === '') {
          cv.style.cursor = 'pointer';
          state.hoveredCell = { subgrid: s, cell: cellIdx };
        } else {
          cv.style.cursor = 'default';
          state.hoveredCell = null;
        }
      } else {
        cv.style.cursor = 'default';
        state.hoveredCell = null;
      }
    });

    wrapElement.addEventListener('mouseleave', () => {
      const state = (parent as any).__visualizer_state__;
      state.hoveredCell = null;
      const cv = wrapElement.querySelector('canvas');
      if (cv) cv.style.cursor = 'default';
    });

    // E. Register standard framework playback handlers
    registerPlaybackHandlers({
      onPlay: () => {
        const state = (parent as any).__visualizer_state__;
        state.isPlaying = true;
        updatePlayBtnIcon(parent, true);
      },
      onPause: () => {
        const state = (parent as any).__visualizer_state__;
        state.isPlaying = false;
        updatePlayBtnIcon(parent, false);
      },
    });

    // F. Tabs switching event listeners
    const tabBtns = container.querySelectorAll('.utt-tab-btn');
    const tabContents = container.querySelectorAll('.utt-tab-content');
    tabBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        tabBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');

        tabContents.forEach((content) => {
          const id = content.getAttribute('id');
          if (id === `tab-${targetTab}`) {
            content.classList.add('active');
          } else {
            content.classList.remove('active');
          }
        });
      });
    });

    // G. Mini Live Feed event delegation listener
    const feedContainer = container.querySelector('.utt-mini-history-feed') as HTMLDivElement;
    feedContainer.addEventListener('click', (e) => {
      const item = (e.target as HTMLElement).closest('.utt-feed-item') as HTMLElement;
      if (item && item.hasAttribute('data-step')) {
        setPlaying(false);
        setStep(parseInt(item.getAttribute('data-step') || '0'));
      }
    });

    stateObj.initialized = true;
  }

  // 4. Update state variables
  stateObj.activeStep = step;

  // 5. Parse current state observation
  const parsedObs = getObservationAtStep(step, steps);

  // 6. Update HTML elements dynamically
  const container = parent.querySelector('.renderer-container') as HTMLDivElement;

  // Odometer & Slider
  const odometer = container.querySelector('.utt-odometer') as HTMLSpanElement;
  const pad = (num: number) => String(num).padStart(2, '0');
  odometer.textContent = `${pad(step)} / ${pad(steps.length - 1)}`;

  const slider = container.querySelector('.utt-slider') as HTMLInputElement;
  slider.value = String(step);

  // Scoreboard wins
  const p0Wins = parsedObs ? parsedObs.subgrid_winners.filter((w: string) => w === 'x').length : 0;
  const p1Wins = parsedObs ? parsedObs.subgrid_winners.filter((w: string) => w === 'o').length : 0;

  // Move counts
  let p0Moves = 0;
  let p1Moves = 0;
  for (let t = 2; t <= step; t++) {
    const act0 = steps[t]?.[0]?.action;
    const act1 = steps[t]?.[1]?.action;
    if (act0 && (act0.actionString || act0.submission !== -1)) {
      p0Moves++;
    } else if (act1 && (act1.actionString || act1.submission !== -1)) {
      p1Moves++;
    }
  }

  // Plaques
  const plaques = container.querySelectorAll('.utt-turn-plaque');
  const isTerminal = parsedObs?.is_terminal || false;
  const activeIdx = isTerminal
    ? -1
    : parsedObs?.current_player === 'x'
      ? 0
      : parsedObs?.current_player === 'o'
        ? 1
        : -1;

  plaques[0].className = `utt-turn-plaque p0 player-card ${activeIdx === 0 ? 'active' : ''}`;
  plaques[0].querySelector('.stats')!.innerHTML = `Moves: ${p0Moves} | Wins: ${p0Wins}`;

  plaques[1].className = `utt-turn-plaque p1 player-card ${activeIdx === 1 ? 'active' : ''}`;
  plaques[1].querySelector('.stats')!.innerHTML = `Moves: ${p1Moves} | Wins: ${p1Wins}`;

  // Log List items active style
  const logList = container.querySelector('.utt-log-list') as HTMLDivElement;
  logList.querySelectorAll('.utt-log-item').forEach((item) => {
    const itemStep = parseInt(item.getAttribute('data-step') || '0');
    if (itemStep === step) {
      item.classList.add('active');
      item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } else {
      item.classList.remove('active');
    }
  });

  // Update Live Move Feed (most recent 5 moves relative to current step)
  const feedContainer = container.querySelector('.utt-mini-history-feed') as HTMLDivElement;
  if (feedContainer) {
    const feedHtml: string[] = [];
    for (let t = step - 2; t <= step + 2; t++) {
      if (t < 0 || t >= steps.length) {
        feedHtml.push(`<div class="utt-feed-item empty"><span class="text">—</span></div>`);
      } else if (t === 0) {
        feedHtml.push(`
          <div class="utt-feed-item ${t === step ? 'active' : ''}" data-step="0">
            <span class="idx">—</span>
            <span class="symbol"></span>
            <span class="text">Match Started</span>
          </div>
        `);
      } else if (t === 1) {
        feedHtml.push(`
          <div class="utt-feed-item ${t === step ? 'active' : ''}" data-step="1">
            <span class="idx">—</span>
            <span class="symbol"></span>
            <span class="text">Sub-grid Selection</span>
          </div>
        `);
      } else {
        const move = stateObj.moves.find((m) => m.stepIndex === t);
        if (move) {
          feedHtml.push(`
            <div class="utt-feed-item ${t === step ? 'active' : ''}" data-step="${t}">
              <span class="idx">${move.index}.</span>
              <span class="symbol ${move.player}">${move.player === 'x' ? '✕' : '◯'}</span>
              <span class="text">${move.text}</span>
            </div>
          `);
        } else {
          feedHtml.push(`<div class="utt-feed-item empty"><span class="text">—</span></div>`);
        }
      }
    }
    feedContainer.innerHTML = feedHtml.join('');
  }

  // Score text in summary
  const scoreRowP0 = container.querySelector('.utt-score-row.p0') as HTMLDivElement;
  const scoreRowP1 = container.querySelector('.utt-score-row.p1') as HTMLDivElement;
  scoreRowP0.textContent = `${player1Name}: ${p0Wins} sub-grid wins`;
  scoreRowP1.textContent = `${player2Name}: ${p1Wins} sub-grid wins`;

  // Mini summary grid
  const miniGrid = container.querySelector('.utt-mini-grid') as HTMLDivElement;
  const activeSubgrid = parsedObs?.active_subgrid;
  miniGrid.innerHTML = Array.from({ length: 9 }, (_, s) => {
    const winner = parsedObs?.subgrid_winners?.[s] || '';
    const isActive = activeSubgrid === null || activeSubgrid === undefined ? winner === '' : s === activeSubgrid;
    let className = 'utt-mini-cell';
    let content = '';
    if (winner === 'x') {
      className += ' x';
      content = '✕';
    } else if (winner === 'o') {
      className += ' o';
      content = '◯';
    } else if (winner === 'draw') {
      className += ' draw';
      content = '—';
    }
    if (isActive && !isTerminal) {
      className += ' active';
    }
    return `<div class="${className}">${content}</div>`;
  }).join('');

  // Contextual Help Box & Winner Overlay
  const contextHelp = container.querySelector('.utt-context-help') as HTMLDivElement;
  let winnerOverlay = container.querySelector('.utt-winner-overlay') as HTMLDivElement;
  if (winnerOverlay) {
    winnerOverlay.remove();
  }

  if (isTerminal) {
    const winner = parsedObs.winner;
    let winnerLabel = 'Match Ended in a Draw';
    let subtitle = 'Both players fought valiantly.';
    let overlayClass = 'draw';
    if (winner === 'x') {
      winnerLabel = `${player1Name} Wins!`;
      subtitle = 'Player 1 (X) successfully conquered the macro-grid!';
      overlayClass = 'x';
    } else if (winner === 'o') {
      winnerLabel = `${player2Name} Wins!`;
      subtitle = 'Player 2 (O) successfully conquered the macro-grid!';
      overlayClass = 'o';
    }

    winnerOverlay = document.createElement('div');
    winnerOverlay.className = `utt-winner-overlay ${overlayClass}`;
    winnerOverlay.innerHTML = `
      <div class="utt-winner-card">
        <div class="utt-winner-title">${winnerLabel}</div>
        <div class="utt-winner-subtitle">${subtitle}</div>
      </div>
    `;
    container.querySelector('.utt-left-pane')!.appendChild(winnerOverlay);

    contextHelp.innerHTML = `<strong>Match Complete!</strong> ${winner === 'draw' ? 'The match ended in a draw.' : `Congratulations to <strong>${winner === 'x' ? player1Name : player2Name}</strong>, who has won the match!`}`;
  } else if (step === 0) {
    contextHelp.innerHTML = `<strong>New Match Started:</strong> <strong>${player1Name} (X)</strong> must select any open sub-grid to make their first move.`;
  } else if (step === 1) {
    contextHelp.innerHTML = `<strong>Sub-grid Selection:</strong> <strong>${player1Name} (X)</strong> is choosing which sub-grid to place their mark in.`;
  } else {
    // Determine the last move text
    const lastMove = stateObj.moves.find((m) => m.stepIndex === step);
    const nextPlayerLabel = activeIdx === 0 ? `${player1Name} (X)` : `${player2Name} (O)`;

    if (lastMove) {
      if (lastMove.cellIdx !== null) {
        const subR = Math.floor(lastMove.subgridIdx / 3);
        const subC = lastMove.subgridIdx % 3;
        const cellR = Math.floor(lastMove.cellIdx / 3);
        const cellC = lastMove.cellIdx % 3;

        // Check if targeted sub-grid is complete
        const targetSubIdx = lastMove.cellIdx;
        const targetSubR = Math.floor(targetSubIdx / 3);
        const targetSubC = targetSubIdx % 3;
        const targetWinner = parsedObs?.subgrid_winners?.[targetSubIdx] || '';

        if (targetWinner !== '') {
          contextHelp.innerHTML = `<strong>Move ${lastMove.index}:</strong> ${lastMove.playerLabel} placed in cell [R${cellR}, C${cellC}] of Sub-grid [R${subR}, C${subC}]. Because the targeted Sub-grid [R${targetSubR}, C${targetSubC}] is already completed, <strong>${nextPlayerLabel}</strong> has a <strong>free move</strong> and can choose any open sub-grid! <strong>(Next Turn: ${nextPlayerLabel})</strong>`;
        } else {
          contextHelp.innerHTML = `<strong>Move ${lastMove.index}:</strong> ${lastMove.playerLabel} placed in cell [R${cellR}, C${cellC}] of Sub-grid [R${subR}, C${subC}]. This mandates that <strong>${nextPlayerLabel}</strong> must play their next move inside Sub-grid <strong>[R${targetSubR}, C${targetSubC}]</strong> (corresponding to that cell index). <strong>(Next Turn: ${nextPlayerLabel})</strong>`;
        }
      } else {
        contextHelp.innerHTML = `<strong>Move ${lastMove.index}:</strong> ${lastMove.text}. Now, they must play a cell inside that sub-grid. <strong>(Next Turn: ${nextPlayerLabel})</strong>`;
      }
    } else {
      contextHelp.innerHTML = `<strong>Game ongoing:</strong> It is currently <strong>${nextPlayerLabel}</strong>'s turn.`;
    }
  }

  // 7. Render Board Canvas
  const wrap = container.querySelector('.board-wrap') as HTMLDivElement;
  let canvas = wrap.querySelector('canvas') as HTMLCanvasElement;
  if (!canvas) {
    canvas = document.createElement('canvas');
    wrap.appendChild(canvas);
  }

  // Track subgrid wins transitions for animation
  if (parsedObs && parsedObs.subgrid_winners) {
    if (stateObj.lastStepWithWinners !== step) {
      stateObj.prevWinners =
        step > 0
          ? JSON.parse(steps[step - 1]?.[0]?.observation?.observationString || '{}').subgrid_winners ||
            Array(9).fill('')
          : Array(9).fill('');
      stateObj.animationStartTime = Date.now();
      stateObj.lastStepWithWinners = step;
    }
  }

  const sizeAndDraw = () => {
    if (stateObj.activeStep !== step) return; // Stale draw call
    const wrapRect = wrap.getBoundingClientRect();
    const cssW = Math.max(1, Math.floor(wrapRect.width));
    const cssH = Math.max(1, Math.floor(wrapRect.height));

    if (canvas.width !== cssW || canvas.height !== cssH) {
      canvas.width = cssW;
      canvas.height = cssH;
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
    }

    const c = canvas.getContext('2d');
    if (!c) return;

    c.clearRect(0, 0, cssW, cssH);

    if (parsedObs && parsedObs.board) {
      const size = Math.min(cssW, cssH) - 40;
      const boardLeft = (cssW - size) / 2;
      const boardTop = Math.min(20, (cssH - size) / 2);
      const subgridSize = size / 3;
      const cellSize = subgridSize / 3;

      const active = parsedObs.active_subgrid;
      const isTerminal = parsedObs.is_terminal;

      // 1. Draw glowing active subgrid backgrounds and frosted glass inactive overlays
      for (let s = 0; s < 9; s++) {
        const majorRow = Math.floor(s / 3);
        const majorCol = s % 3;
        const x = boardLeft + majorCol * subgridSize;
        const y = boardTop + majorRow * subgridSize;
        const winner = parsedObs.subgrid_winners?.[s] || '';

        const isPlayable = !isTerminal && (active === null || active === undefined ? winner === '' : s === active);

        if (isPlayable) {
          c.save();
          // Glowing active perimeter pulse
          const time = Date.now() / 300;
          const pulse = 2 + Math.sin(time) * 1.5;

          c.shadowColor = '#eab308'; // Gold glow
          c.shadowBlur = 10 + pulse;
          c.strokeStyle = '#eab308';
          c.lineWidth = 3;
          c.strokeRect(x + 5, y + 5, subgridSize - 10, subgridSize - 10);

          // Faint golden interior
          c.fillStyle = 'rgba(254, 240, 138, 0.15)';
          c.fillRect(x + 5, y + 5, subgridSize - 10, subgridSize - 10);
          c.restore();
        } else if (!isTerminal && winner === '') {
          // Frosted glass overlay on non-active but playable subgrids
          c.save();
          c.fillStyle = 'rgba(241, 245, 249, 0.45)'; // Frosted white overlay
          c.fillRect(x + 4, y + 4, subgridSize - 8, subgridSize - 8);
          c.restore();
        }
      }

      // 2. Draw minor grid lines (etched line look)
      c.save();
      c.strokeStyle = 'rgba(100, 116, 139, 0.25)'; // Slate 500 faint
      c.lineWidth = 1;
      for (let majorRow = 0; majorRow < 3; majorRow++) {
        for (let majorCol = 0; majorCol < 3; majorCol++) {
          const subgridX = boardLeft + majorCol * subgridSize;
          const subgridY = boardTop + majorRow * subgridSize;

          for (let r = 1; r < 3; r++) {
            c.beginPath();
            c.moveTo(subgridX + 12, subgridY + r * cellSize);
            c.lineTo(subgridX + subgridSize - 12, subgridY + r * cellSize);
            c.stroke();
          }

          for (let col = 1; col < 3; col++) {
            c.beginPath();
            c.moveTo(subgridX + col * cellSize, subgridY + 12);
            c.lineTo(subgridX + col * cellSize, subgridY + subgridSize - 12);
            c.stroke();
          }
        }
      }
      c.restore();

      // 3. Draw major grid lines (charcoal brushed look)
      c.save();
      c.strokeStyle = '#334155'; // Slate 700
      c.lineWidth = 4;
      c.lineCap = 'round';
      c.shadowColor = 'rgba(15, 23, 42, 0.15)';
      c.shadowBlur = 4;
      c.shadowOffsetY = 2;
      for (let i = 1; i < 3; i++) {
        // Horizontal
        c.beginPath();
        c.moveTo(boardLeft, boardTop + i * subgridSize);
        c.lineTo(boardLeft + size, boardTop + i * subgridSize);
        c.stroke();

        // Vertical
        c.beginPath();
        c.moveTo(boardLeft + i * subgridSize, boardTop);
        c.lineTo(boardLeft + i * subgridSize, boardTop + size);
        c.stroke();
      }
      c.restore();

      // 4. Pre-calculate cell move numbers
      const moveOrders = calculateMoveOrders(step, steps);

      // 5. Draw cell marks & subgrid winners
      let needsRefigStep = false;
      for (let s = 0; s < 9; s++) {
        const majorRow = Math.floor(s / 3);
        const majorCol = s % 3;
        const subgridX = boardLeft + majorCol * subgridSize;
        const subgridY = boardTop + majorRow * subgridSize;

        const subWinner = parsedObs.subgrid_winners?.[s] || '';
        const prevWinner = stateObj.prevWinners?.[s] || '';

        if (subWinner !== '') {
          // Subgrid is won! Draw large ornate winning mark
          const centerX = subgridX + subgridSize / 2;
          const centerY = subgridY + subgridSize / 2;
          const radius = subgridSize * 0.35;

          c.save();

          // Animate scale if it was just won
          let scale = 1;
          if (subWinner !== prevWinner) {
            const elapsed = Date.now() - stateObj.animationStartTime;
            scale = Math.min(1, elapsed / 400); // 400ms transition
            if (scale < 1) {
              needsRefigStep = true;
            }
          }

          c.translate(centerX, centerY);
          c.scale(scale, scale);
          c.translate(-centerX, -centerY);

          if (subWinner === 'x') {
            c.fillStyle = 'rgba(37, 99, 235, 0.08)'; // Sapphire faint glow
            c.fillRect(subgridX + 4, subgridY + 4, subgridSize - 8, subgridSize - 8);
            drawMetallicX(c, centerX, centerY, radius, 8, true);
          } else if (subWinner === 'o') {
            c.fillStyle = 'rgba(220, 38, 38, 0.08)'; // Ruby faint glow
            c.fillRect(subgridX + 4, subgridY + 4, subgridSize - 8, subgridSize - 8);
            drawMetallicO(c, centerX, centerY, radius, 8, false);
          } else if (subWinner === 'draw') {
            c.fillStyle = 'rgba(71, 85, 105, 0.15)'; // Tie slate overlay
            c.fillRect(subgridX + 4, subgridY + 4, subgridSize - 8, subgridSize - 8);

            c.strokeStyle = '#64748b';
            c.lineWidth = 4;
            c.lineCap = 'round';
            c.beginPath();
            c.moveTo(centerX - radius, centerY);
            c.lineTo(centerX + radius, centerY);
            c.stroke();
          }

          c.restore();
        } else {
          // Subgrid is ongoing. Draw individual cell marks
          const subBoard = parsedObs.board[s];
          for (let cellIdx = 0; cellIdx < 9; cellIdx++) {
            const minorRow = Math.floor(cellIdx / 3);
            const minorCol = cellIdx % 3;
            const cellX = subgridX + minorCol * cellSize + cellSize / 2;
            const cellY = subgridY + minorRow * cellSize + cellSize / 2;

            const mark = subBoard[cellIdx];
            const radius = cellSize * 0.28;

            if (mark === 'x') {
              drawMetallicX(c, cellX, cellY, radius, 3.5, true);
            } else if (mark === 'o') {
              drawMetallicO(c, cellX, cellY, radius, 3.5, false);
            }

            // Draw move numbers
            const moveNo = moveOrders[s][cellIdx];
            if (moveNo > 0) {
              c.save();
              c.fillStyle = '#64748b'; // Slate 500
              c.font = 'bold 8.5px "Share Tech Mono", monospace';
              c.textAlign = 'right';
              c.textBaseline = 'bottom';
              c.fillText(
                `${mark.toUpperCase()}${moveNo}`,
                subgridX + minorCol * cellSize + cellSize - 3,
                subgridY + minorRow * cellSize + cellSize - 2
              );
              c.restore();
            }
          }
        }
      }

      // 6. Draw hovered cell preview translucent mark
      if (stateObj.hoveredCell && !isTerminal) {
        const { subgrid: s, cell: cellIdx } = stateObj.hoveredCell;
        const majorRow = Math.floor(s / 3);
        const majorCol = s % 3;
        const subgridX = boardLeft + majorCol * subgridSize;
        const subgridY = boardTop + majorRow * subgridSize;
        const minorRow = Math.floor(cellIdx / 3);
        const minorCol = cellIdx % 3;
        const cellX = subgridX + minorCol * cellSize + cellSize / 2;
        const cellY = subgridY + minorRow * cellSize + cellSize / 2;

        const radius = cellSize * 0.28;
        const turnPlayer = parsedObs.current_player;

        c.save();
        // Inset cell shadow background
        c.fillStyle = 'rgba(56, 189, 248, 0.12)';
        c.fillRect(subgridX + minorCol * cellSize + 2, subgridY + minorRow * cellSize + 2, cellSize - 4, cellSize - 4);
        c.strokeStyle = 'rgba(56, 189, 248, 0.4)';
        c.lineWidth = 1.5;
        c.strokeRect(
          subgridX + minorCol * cellSize + 2,
          subgridY + minorRow * cellSize + 2,
          cellSize - 4,
          cellSize - 4
        );

        // Translucent mark
        if (turnPlayer === 'x') {
          drawMetallicX(c, cellX, cellY, radius, 3, true);
          // Apply a fade-out opacity overlay by drawing white with opacity
          c.fillStyle = 'rgba(248, 250, 252, 0.5)';
          c.beginPath();
          c.arc(cellX, cellY, radius + 2, 0, 2 * Math.PI);
          c.fill();
        } else if (turnPlayer === 'o') {
          drawMetallicO(c, cellX, cellY, radius, 3, false);
          c.fillStyle = 'rgba(248, 250, 252, 0.5)';
          c.beginPath();
          c.arc(cellX, cellY, radius + 2, 0, 2 * Math.PI);
          c.fill();
        }
        c.restore();
      }

      // 7. Request another frame if we are playing or in transition
      if (stateObj.isPlaying || needsRefigStep) {
        requestAnimationFrame(sizeAndDraw);
      }
    }
  };

  requestAnimationFrame(sizeAndDraw);
}

// Helpers
function getObservationAtStep(stepIndex: number, steps: any[]): any {
  const currentStep = steps[stepIndex];
  const rawObs = currentStep?.[0]?.observation?.observationString;
  if (rawObs) {
    try {
      return JSON.parse(rawObs);
    } catch {
      return null;
    }
  }
  return null;
}

function updatePlayBtnIcon(parent: HTMLElement, isPlaying: boolean) {
  const icon = parent.querySelector('.utt-btn-play i');
  if (icon) {
    icon.textContent = isPlaying ? 'pause' : 'play_arrow';
  }
}

function calculateMoveOrders(step: number, steps: any[]): number[][] {
  const moveOrders = Array.from({ length: 9 }, () => Array(9).fill(0));
  let cellMoveCount = 0;
  for (let t = 2; t <= step; t++) {
    const act0 = steps[t]?.[0]?.action;
    const act1 = steps[t]?.[1]?.action;
    let actionString = '';
    if (act0 && (act0.actionString || act0.submission !== -1)) {
      actionString = act0.actionString || '';
    } else if (act1 && (act1.actionString || act1.submission !== -1)) {
      actionString = act1.actionString || '';
    }

    if (actionString && !actionString.startsWith('Choose local board')) {
      const match = actionString.match(/Local board (\d+): [xo]\((\d+),(\d+)\)/);
      if (match) {
        const subgridIdx = parseInt(match[1]);
        const cellRow = parseInt(match[2]);
        const cellCol = parseInt(match[3]);
        const cellIdx = cellRow * 3 + cellCol;
        cellMoveCount++;
        moveOrders[subgridIdx][cellIdx] = cellMoveCount;
      }
    }
  }
  return moveOrders;
}

function drawX(c: CanvasRenderingContext2D, x: number, y: number, r: number, lineWidth: number, color: string) {
  c.save();
  c.strokeStyle = color;
  c.lineWidth = lineWidth;
  c.lineCap = 'round';
  c.beginPath();
  c.moveTo(x - r, y - r);
  c.lineTo(x + r, y + r);
  c.moveTo(x + r, y - r);
  c.lineTo(x - r, y + r);
  c.stroke();
  c.restore();
}

function drawO(c: CanvasRenderingContext2D, x: number, y: number, r: number, lineWidth: number, color: string) {
  c.save();
  c.strokeStyle = color;
  c.lineWidth = lineWidth;
  c.beginPath();
  c.arc(x, y, r, 0, 2 * Math.PI);
  c.stroke();
  c.restore();
}

function drawMetallicX(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  lineWidth: number,
  isSapphire: boolean
) {
  c.save();
  c.lineCap = 'round';

  // Drop shadow
  c.strokeStyle = 'rgba(15, 23, 42, 0.45)';
  c.lineWidth = lineWidth;
  c.beginPath();
  c.moveTo(x - r + 1, y - r + 1.5);
  c.lineTo(x + r + 1, y + r + 1.5);
  c.moveTo(x + r + 1, y - r + 1.5);
  c.lineTo(x - r + 1, y + r + 1.5);
  c.stroke();

  // Metallic gradient
  const grad = c.createLinearGradient(x - r, y - r, x + r, y + r);
  if (isSapphire) {
    grad.addColorStop(0, '#60a5fa');
    grad.addColorStop(0.5, '#2563eb');
    grad.addColorStop(1, '#1e3a8a');
  } else {
    grad.addColorStop(0, '#f87171');
    grad.addColorStop(0.5, '#dc2626');
    grad.addColorStop(1, '#7f1d1d');
  }

  c.strokeStyle = grad;
  c.lineWidth = lineWidth;
  c.beginPath();
  c.moveTo(x - r, y - r);
  c.lineTo(x + r, y + r);
  c.moveTo(x + r, y - r);
  c.lineTo(x - r, y + r);
  c.stroke();

  // Highlight
  c.strokeStyle = isSapphire ? 'rgba(219, 234, 254, 0.5)' : 'rgba(fee2e2, 0.5)';
  c.lineWidth = lineWidth * 0.3;
  c.beginPath();
  c.moveTo(x - r - 0.5, y - r - 0.5);
  c.lineTo(x + r - 0.5, y + r - 0.5);
  c.moveTo(x + r - 0.5, y - r - 0.5);
  c.lineTo(x - r - 0.5, y + r - 0.5);
  c.stroke();

  c.restore();
}

function drawMetallicO(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  lineWidth: number,
  isSapphire: boolean
) {
  c.save();

  // Drop shadow
  c.strokeStyle = 'rgba(15, 23, 42, 0.45)';
  c.lineWidth = lineWidth;
  c.beginPath();
  c.arc(x + 1, y + 1.5, r, 0, 2 * Math.PI);
  c.stroke();

  // Metallic gradient
  const grad = c.createLinearGradient(x - r, y - r, x + r, y + r);
  if (isSapphire) {
    grad.addColorStop(0, '#60a5fa');
    grad.addColorStop(0.5, '#2563eb');
    grad.addColorStop(1, '#1e3a8a');
  } else {
    grad.addColorStop(0, '#f87171');
    grad.addColorStop(0.5, '#dc2626');
    grad.addColorStop(1, '#7f1d1d');
  }

  c.strokeStyle = grad;
  c.lineWidth = lineWidth;
  c.beginPath();
  c.arc(x, y, r, 0, 2 * Math.PI);
  c.stroke();

  // Highlight
  c.strokeStyle = isSapphire ? 'rgba(219, 234, 254, 0.5)' : 'rgba(fee2e2, 0.5)';
  c.lineWidth = lineWidth * 0.3;
  c.beginPath();
  c.arc(x - 0.5, y - 0.5, r, 0, 2 * Math.PI);
  c.stroke();

  c.restore();
}

function drawMiniBoard(canvas: HTMLCanvasElement, stepIndex: number, steps: any[]) {
  const c = canvas.getContext('2d');
  if (!c) return;

  const width = canvas.width;
  const height = canvas.height;
  c.clearRect(0, 0, width, height);

  let state: any = null;
  const rawObs = steps[stepIndex]?.[0]?.observation?.observationString;
  if (rawObs) {
    try {
      state = JSON.parse(rawObs);
    } catch {}
  }
  if (!state || !state.board) return;

  const size = Math.min(width, height) - 6;
  const boardLeft = (width - size) / 2;
  const boardTop = (height - size) / 2;
  const subgridSize = size / 3;
  const cellSize = subgridSize / 3;

  for (let s = 0; s < 9; s++) {
    const majorRow = Math.floor(s / 3);
    const majorCol = s % 3;
    const subgridX = boardLeft + majorCol * subgridSize;
    const subgridY = boardTop + majorRow * subgridSize;
    const subWinner = state.subgrid_winners?.[s] || '';

    if (subWinner === 'x') {
      c.fillStyle = 'rgba(37, 99, 235, 0.12)';
      c.fillRect(subgridX, subgridY, subgridSize, subgridSize);
    } else if (subWinner === 'o') {
      c.fillStyle = 'rgba(220, 38, 38, 0.12)';
      c.fillRect(subgridX, subgridY, subgridSize, subgridSize);
    }
  }

  c.strokeStyle = '#475569';
  c.lineWidth = 1.5;
  for (let i = 1; i < 3; i++) {
    c.beginPath();
    c.moveTo(boardLeft, boardTop + i * subgridSize);
    c.lineTo(boardLeft + size, boardTop + i * subgridSize);
    c.stroke();

    c.beginPath();
    c.moveTo(boardLeft + i * subgridSize, boardTop);
    c.lineTo(boardLeft + i * subgridSize, boardTop + size);
    c.stroke();
  }

  for (let s = 0; s < 9; s++) {
    const majorRow = Math.floor(s / 3);
    const majorCol = s % 3;
    const subgridX = boardLeft + majorCol * subgridSize;
    const subgridY = boardTop + majorRow * subgridSize;

    const subWinner = state.subgrid_winners?.[s] || '';
    if (subWinner === 'x' || subWinner === 'o') {
      const centerX = subgridX + subgridSize / 2;
      const centerY = subgridY + subgridSize / 2;
      const r = subgridSize * 0.35;
      if (subWinner === 'x') {
        drawX(c, centerX, centerY, r, 2.5, 'rgba(37, 99, 235, 0.7)');
      } else {
        drawO(c, centerX, centerY, r, 2.5, 'rgba(220, 38, 38, 0.7)');
      }
    } else {
      const subBoard = state.board[s];
      for (let cellIdx = 0; cellIdx < 9; cellIdx++) {
        const minorRow = Math.floor(cellIdx / 3);
        const minorCol = cellIdx % 3;
        const cellX = subgridX + minorCol * cellSize + cellSize / 2;
        const cellY = subgridY + minorRow * cellSize + cellSize / 2;
        const mark = subBoard[cellIdx];
        const r = cellSize * 0.28;

        if (mark === 'x') {
          drawX(c, cellX, cellY, r, 1.2, '#2563eb');
        } else if (mark === 'o') {
          drawO(c, cellX, cellY, r, 1.2, '#dc2626');
        }
      }
    }
  }
}
