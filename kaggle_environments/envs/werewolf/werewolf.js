function renderer({
  environment,
  step,
  parent,
  height = 700, // Default height
  width = 1100, // Default width
}) {
  // --- CSS for the UI (remains the same) ---
  const css = `
        :root {
            --night-bg: #2c3e50;
            --day-bg: #3498db;
            --night-text: #ecf0f1;
            --day-text: #2c3e50;
            --dead-filter: grayscale(100%) brightness(50%);
            --active-border: #f1c40f;
        }
        .werewolf-parent {
            position: relative;
            overflow: hidden;
            width: 100%;
            height: 100%;
        }
        .background {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            background-size: cover;
            background-position: center;
            transition: background-image 1s ease-in-out;
        }
//        .werewolf-parent.night .background {
//            background-image: url('./kaggle_environments/envs/werewolf/game/_ui/img/background-night.png');
//        }
//        .werewolf-parent.day .background {
//            background-image: url('./kaggle_environments/envs/werewolf/game/_ui/img/background-day.png');
//        }
        .main-container {
            position: relative;
            z-index: 1;
            display: flex;
            height: 100%;
            width: 100%;
            background-color: rgba(0,0,0,0.3);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--night-text);
        }
        .left-panel {
            width: 300px;
            flex-shrink: 0;
            background-color: rgba(44, 62, 80, 0.85);
            padding: 15px;
            display: flex;
            flex-direction: column;
            height: 100%;
            box-sizing: border-box;
        }
        .right-panel {
            flex-grow: 1;
            background-color: rgba(44, 62, 80, 0.85);
            padding: 15px;
            display: flex;
            flex-direction: column;
            height: 100%;
            box-sizing: border-box;
        }
        .right-panel h1, #player-list-area h1 {
            margin-top: 0;
            text-align: center;
            font-size: 1.5em;
            color: var(--night-text);
            border-bottom: 2px solid var(--night-text);
            padding-bottom: 10px;
            flex-shrink: 0;
        }
        #player-list-area {
            flex: 3;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        #player-list-container {
            overflow-y: auto;
            scrollbar-width: thin;
            flex-grow: 1;
        }
        #player-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .player-card {
            display: flex;
            align-items: center;
            background-color: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 5px solid transparent;
            transition: all 0.3s ease;
        }
        .player-card.active {
            border-left-color: var(--active-border);
            box-shadow: 0 0 15px rgba(241, 196, 15, 0.5);
        }
        .player-card.dead {
            opacity: 0.6;
        }
        .player-card .avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            margin-right: 15px;
            object-fit: cover;
            flex-shrink: 0;
            background-color: #fff;
        }
        .player-card.dead .avatar {
             filter: var(--dead-filter);
        }
        .player-info {
            flex-grow: 1;
            overflow: hidden;
        }
        .player-name {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 3px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .player-role, .player-status {
            font-size: 0.9em;
            color: #bdc3c7;
        }
        #chat-log {
            list-style: none;
            padding: 0;
            margin: 0;
            flex-grow: 1;
            overflow-y: auto;
            scrollbar-width: thin;
        }
        .chat-entry {
            display: flex;
            margin-bottom: 15px;
            align-items: flex-start;
        }
        .chat-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px;
            object-fit: cover;
            flex-shrink: 0;
        }
        .message-content {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }
        .balloon {
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
            transition: background-color 0.3s ease;
        }
        .chat-entry.event-day .balloon {
            background-color: rgba(236, 240, 241, 0.1);
            color: var(--night-text);
        }
        .chat-entry.event-night .balloon {
            background-color: rgba(0, 0, 0, 0.25);
        }
        .msg-entry {
            border-left: 3px solid #f39c12;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        .msg-entry.event-day {
            background-color: rgba(236, 240, 241, 0.1); /* Lighter background for Day events */
        }
        .msg-entry.event-night {
            background-color: rgba(0, 0, 0, 0.25); /* Darker background for Night events */
        }
        .reasoning-text {
            font-size: 0.85em;
            color: #bdc3c7;
            font-style: italic;
            margin-top: 5px;
            padding-left: 10px;
            border-left: 2px solid #555;
        }
        .msg-entry.game-event {
             border-left-color: #e74c3c;
        }
        .msg-entry.game-win {
             border-left-color: #2ecc71;
        }
        #chat-log cite {
            font-style: normal;
            font-weight: bold;
            display: block;
            font-size: 0.9em;
            color: #ecf0f1;
            margin-bottom: 5px;
        }
        .moderator-announcement {
            margin: 10px 0;
        }
        .moderator-announcement-content {
            padding: 10px;
            border-radius: 5px;
            transition: background-color 0.3s ease;
            border-left: 5px solid #2ecc71;
            color: var(--night-text);
        }
        .moderator-announcement-content.event-day {
            background-color: rgba(236, 240, 241, 0.1);
        }
        .moderator-announcement-content.event-night {
            background-color: rgba(0, 0, 0, 0.25);
        }
        .phase-separator {
            display: flex;
            align-items: center;
            text-align: center;
            margin: 20px 0 15px;
            color: #bdc3c7;
            font-variant: small-caps;
        }
        .phase-separator::before,
        .phase-separator::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid #555;
        }
        .phase-separator:not(:empty)::before {
            margin-right: 1em;
        }
        .phase-separator:not(:empty)::after {
            margin-left: 1em;
        }
        .phase-separator span {
            padding: 0 10px;
            font-style: italic;
            font-size: 1em;
            letter-spacing: 1px;
        }
        .timestamp {
            font-size: 0.8em;
            color: #bdc3c7;
            margin-left: 10px;
            font-weight: normal;
        }
    `;

  // --- Helper Functions ---
  function formatTimestamp(isoString) {
    if (!isoString) return '';
    try {
        return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch (e) {
        return '';
    }
  }

  function renderPlayerList(container, gameState, actingPlayerName) {
    container.innerHTML = '<h1>Players</h1>';
    const listContainer = document.createElement('div');
    listContainer.id = 'player-list-container';
    const playerUl = document.createElement('ul');
    playerUl.id = 'player-list';

    gameState.players.forEach(player => {
        const li = document.createElement('li');
        li.className = 'player-card';
        if (!player.is_alive) li.classList.add('dead');
        if (player.name === actingPlayerName) li.classList.add('active');

        const roleText = player.role !== 'Unknown' ? `Role: ${player.role}` : 'Role: Unknown';

        li.innerHTML = `
            <img src="${player.thumbnail}" alt="${player.name}" class="avatar">
            <div class="player-info">
                <div class="player-name" title="${player.name}">${player.name}</div>
                <div class="player-role">${roleText}</div>
            </div>
        `;
        playerUl.appendChild(li);
    });

    listContainer.appendChild(playerUl);
    container.appendChild(listContainer);
  }

  function renderEventLog(container, gameState, playerMap) {
    container.innerHTML = '<h1>Event Log</h1>';
    const logUl = document.createElement('ul');
    logUl.id = 'chat-log';

    const logEntries = gameState.eventLog;

    if (logEntries.length === 0) {
        const li = document.createElement('li');
        li.className = 'msg-entry';
        li.innerHTML = `<cite>System</cite><div>The game is about to begin...</div>`;
        logUl.appendChild(li);
    } else {
        let lastPhaseIdentifier = null;

        logEntries.forEach(entry => {
            const li = document.createElement('li');
            let reasoningHtml = entry.reasoning ? `<div class="reasoning-text">"${entry.reasoning}"</div>` : '';
            const phaseClass = `event-${(entry.phase || 'day').toLowerCase()}`;
            const timestampHtml = `<span class="timestamp">${formatTimestamp(entry.timestamp)}</span>`;

            if (entry.day !== Infinity) {
                const phase = (entry.phase || 'Day').toUpperCase();
                const currentPhaseIdentifier = `${phase} ${entry.day}`;
                if (currentPhaseIdentifier !== lastPhaseIdentifier) {
                    const separator = document.createElement('li');
                    separator.className = 'phase-separator';
                    separator.innerHTML = `<span>${phase} ${entry.day}</span>`;
                    logUl.appendChild(separator);
                    lastPhaseIdentifier = currentPhaseIdentifier;
                }
            }

            switch (entry.type) {
                case 'chat':
                    const speaker = playerMap.get(entry.speaker);
                    if (!speaker) return;
                    li.className = `chat-entry event-day`;
                    li.innerHTML = `
                        <img src="${speaker.thumbnail}" alt="${speaker.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${speaker.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text"><quote>${entry.message}</quote></div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'seer_inspection':
                    const seerInspector = playerMap.get(entry.actor);
                    if (!seerInspector) return;
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${seerInspector.thumbnail}" alt="${seerInspector.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Private (As Seer) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">Chose to inspect <strong>${entry.target}</strong>'s role.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'seer_inspection_result':
                    const seerResultViewer = playerMap.get(entry.seer);
                    if (!seerResultViewer) return;
                    li.className = `chat-entry ${phaseClass}`;
                    li.innerHTML = `
                        <img src="${seerResultViewer.thumbnail}" alt="${seerResultViewer.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Private (As Seer) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">Saw <strong>${entry.target}</strong>'s role is a <strong>${entry.role}</strong>.</div>
                            </div>
                        </div>
                    `;
                    break;
                case 'doctor_heal_action':
                    const doctor = playerMap.get(entry.actor);
                    if (!doctor) return;
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${doctor.thumbnail}" alt="${doctor.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Private (As Doctor) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">Chose to heal <strong>${entry.target}</strong>.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'system':
                    if (entry.text && entry.text.includes('has begun')) return;
                    li.className = `moderator-announcement`;
                    li.innerHTML = `
                        <cite>Moderator &#128226; ${timestampHtml}</cite>
                        <div class="moderator-announcement-content ${phaseClass}">
                            <div class="msg-text">${entry.text}</div>
                        </div>
                    `;
                    break;
                case 'exile':
                    li.className = `msg-entry game-event event-day`;
                    li.innerHTML = `<cite>Exile ${timestampHtml}</cite><div class="msg-text"><strong>${entry.name}</strong> (${entry.role}) was exiled by vote.</div>`;
                    break;
                case 'elimination':
                    li.className = `msg-entry game-event event-night`;
                    li.innerHTML = `<cite>Elimination ${timestampHtml}</cite><div class="msg-text"><strong>${entry.name}</strong> was eliminated. Their role was a ${entry.role}.</div>`;
                    break;
                case 'save':
                     li.className = `msg-entry event-night`;
                     li.innerHTML = `<cite>Doctor Save ${timestampHtml}</cite><div class="msg-text">Player <strong>${entry.saved_player}</strong> was attacked but saved by a Doctor!</div>`;
                    break;
                case 'vote':
                    const voter = playerMap.get(entry.name);
                    if (!voter) return;
                    li.className = `chat-entry event-day`;
                    li.innerHTML = `
                        <img src="${voter.thumbnail}" alt="${voter.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${voter.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">Votes to eliminate <strong>${entry.target}</strong>.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                     `;
                     break;
                case 'night_vote':
                    const nightVoter = playerMap.get(entry.name);
                    if (!nightVoter) return;
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${nightVoter.thumbnail}" alt="${nightVoter.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Secret Vote (As Werewolf) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">Votes to eliminate <strong>${entry.target}</strong>.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'game_over':
                    li.className = `msg-entry game-win ${phaseClass}`;
                    li.innerHTML = `
                        <cite>Game Over ${timestampHtml}</cite>
                        <div class="msg-text">
                            <div>The <strong>${entry.winner}</strong> team has won!</div><br>
                            <div><strong>Winning Team:</strong> ${entry.winners.join(', ')}</div>
                            <div><strong>Losing Team:</strong> ${entry.losers.join(', ')}</div>
                        </div>
                    `;
                    break;
            }
            if (li.innerHTML) logUl.appendChild(li);
        });
    }

    container.appendChild(logUl);
    logUl.scrollTop = logUl.scrollHeight;
  }

    parent.innerHTML = ''; // Clear previous rendering

    const container = document.createElement("div");
    Object.assign(container.style, {
        fontFamily: "Arial, sans-serif",
        padding: "15px",
        width: `${width}px`,
        height: `${height}px`,
        overflow: "hidden",
        position: 'relative',
    });

    if (!environment || !environment.steps || environment.steps.length === 0 || step >= environment.steps.length) {
        container.textContent = "Waiting for game data or invalid step...";
        parent.appendChild(container);
        return;
    }

    // --- State Reconstruction ---
    let gameState = { players: [], day: 0, phase: 'GAME_SETUP', game_state_phase: 'DAY', gameWinner: null, eventLog: [] };
    const firstObs = environment.steps[0]?.[0]?.observation?.raw_observation;
    if (!firstObs) {
        container.textContent = "Waiting for game data...";
        parent.appendChild(container);
        return;
    }

    const playerThumbnails = firstObs.player_thumbnails || {};
    const allPlayerNamesList = firstObs.all_player_ids;
    gameState.players = allPlayerNamesList.map(name => ({
        name: name, is_alive: true, role: 'Unknown', team: 'Unknown', status: 'Alive',
        thumbnail: playerThumbnails[name] || `https://via.placeholder.com/40/2c3e50/ecf0f1?text=${name.charAt(0)}`
    }));
    const playerMap = new Map(gameState.players.map(p => [p.name, p]));

    // Get initial roles from the start of the game
    const roleAndTeamMap = new Map();
    const moderatorInitialLog = environment.info?.MODERATOR_OBSERVATION?.[0] || [];
    moderatorInitialLog.forEach(dataEntry => {
        if (dataEntry.data_type === 'GameStartRoleDataEntry') {
            const historyEvent = JSON.parse(dataEntry.json_str);
            const data = historyEvent.data;
            if (data) roleAndTeamMap.set(data.player_id, { role: data.role, team: data.team });
        }
    });
    roleAndTeamMap.forEach((info, playerId) => {
        const player = playerMap.get(playerId);
        if (player) { player.role = info.role; player.team = info.team; }
    });

    const processedEvents = new Set();
    let lastPhase = null;
    let lastDay = -1;

    for (let s = 0; s <= step; s++) {
        const stepStateList = environment.steps[s];
        if (!stepStateList) continue;

        // Update overall game state from the first agent in the current step
        const currentObsForStep = stepStateList[0]?.observation?.raw_observation;
        if (currentObsForStep) {
            if (currentObsForStep.day > lastDay) {
                if (currentObsForStep.day > 0) gameState.eventLog.push({ type: 'system', day: currentObsForStep.day, phase: 'DAY', text: `Day ${currentObsForStep.day} has begun.` });
            }
            lastDay = currentObsForStep.day;
            lastPhase = currentObsForStep.phase;
            gameState.day = currentObsForStep.day;
            gameState.phase = currentObsForStep.phase;
            gameState.game_state_phase = currentObsForStep.game_state_phase;
        }

        // Process confirmed events from the moderator log
        const moderatorLogForStep = environment.info?.MODERATOR_OBSERVATION?.[s] || [];
        moderatorLogForStep.forEach(dataEntry => {
             const eventKey = dataEntry.json_str;
             if (processedEvents.has(eventKey)) return;
             processedEvents.add(eventKey);

             const historyEvent = JSON.parse(dataEntry.json_str);
             if (!historyEvent.data) return;
             const data = historyEvent.data;
             const timestamp = historyEvent.created_at;

             switch (dataEntry.data_type) {
                case 'ChatDataEntry':
                    gameState.eventLog.push({ type: 'chat', day: historyEvent.day, phase: historyEvent.phase, speaker: data.actor_id, message: data.message, reasoning: data.reasoning, timestamp });
                    break;
                case 'DayExileVoteDataEntry':
                    gameState.eventLog.push({ type: 'vote', day: historyEvent.day, phase: historyEvent.phase, name: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'WerewolfNightVoteDataEntry':
                    gameState.eventLog.push({ type: 'night_vote', day: historyEvent.day, phase: historyEvent.phase, name: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'DoctorHealActionDataEntry':
                    gameState.eventLog.push({ type: 'doctor_heal_action', day: historyEvent.day, phase: historyEvent.phase, actor: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'SeerInspectActionDataEntry':
                    gameState.eventLog.push({ type: 'seer_inspection', day: historyEvent.day, phase: historyEvent.phase, actor: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'DayExileElectedDataEntry':
                    gameState.eventLog.push({ type: 'exile', day: historyEvent.day, phase: 'DAY', name: data.elected_player_id, role: data.elected_player_role_name, timestamp });
                    break;
                case 'WerewolfNightEliminationDataEntry':
                    gameState.eventLog.push({ type: 'elimination', day: historyEvent.day, phase: 'NIGHT', name: data.eliminated_player_id, role: data.eliminated_player_role_name, timestamp });
                    break;
                case 'SeerInspectResultDataEntry':
                    gameState.eventLog.push({ type: 'seer_inspection_result', day: historyEvent.day, phase: 'NIGHT', seer: data.actor_id, target: data.target_id, role: data.role, timestamp });
                    break;
                case 'DoctorSaveDataEntry':
                    gameState.eventLog.push({ type: 'save', day: historyEvent.day, phase: 'NIGHT', saved_player: data.saved_player_id, timestamp });
                    break;
                case 'GameEndResultsDataEntry':
                    gameState.gameWinner = data.winner_team;
                    const winners = gameState.players.filter(p => p.team === data.winner_team).map(p => p.name);
                    const losers = gameState.players.filter(p => p.team !== data.winner_team).map(p => p.name);
                    gameState.eventLog.push({ type: 'game_over', day: Infinity, phase: 'GAME_OVER', winner: data.winner_team, winners, losers, timestamp });
                    break;
                default:
                    if (historyEvent.entry_type === "moderator_announcement") {
                        gameState.eventLog.push({ type: 'system', day: historyEvent.day, phase: historyEvent.phase, text: historyEvent.description, timestamp});
                    }
                    break;
             }
        });
    }

    // Update player statuses based on the final log
    gameState.players.forEach(p => { p.is_alive = true; p.status = 'Alive'; });
    gameState.eventLog.forEach(entry => {
        if (entry.type === 'exile' || entry.type === 'elimination') {
            const player = playerMap.get(entry.name);
            if (player) {
                player.is_alive = false;
                player.status = entry.type === 'exile' ? 'Exiled' : 'Eliminated';
            }
        }
    });

    gameState.eventLog.sort((a, b) => {
        if (a.day !== b.day) return a.day - b.day;
        const phaseOrder = { 'DAY': 1, 'NIGHT': 2 };
        const aPhase = (a.phase || '').toUpperCase();
        const bPhase = (b.phase || '').toUpperCase();
        const aOrder = phaseOrder[aPhase] || 99;
        const bOrder = phaseOrder[bPhase] || 99;
        if (aOrder !== bOrder) return aOrder - bOrder;
        if (a.timestamp && b.timestamp) return new Date(a.timestamp) - new Date(b.timestamp);
        return 0;
    });

    const lastStepStateList = environment.steps[step];
    const actingPlayerIndex = lastStepStateList.findIndex(s => s.status === 'ACTIVE');
    const actingPlayerName = actingPlayerIndex !== -1 ? allPlayerNamesList[actingPlayerIndex] : "N.A";

    parent.innerHTML = '';
    Object.assign(parent.style, { width: `${width}px`, height: `${height}px` });
    parent.className = 'werewolf-parent';

    const style = document.createElement('style');
    style.textContent = css;
    parent.appendChild(style);

    const isNight = gameState.game_state_phase === 'Night';
    parent.classList.toggle('night', isNight);
    parent.classList.toggle('day', !isNight);

    const background = document.createElement('div');
    background.className = 'background';
    const mainContainer = document.createElement('div');
    mainContainer.className = 'main-container';
    parent.appendChild(background);
    parent.appendChild(mainContainer);

    const leftPanel = document.createElement('div');
    leftPanel.className = 'left-panel';
    mainContainer.appendChild(leftPanel);
    const playerListArea = document.createElement('div');
    playerListArea.id = 'player-list-area';
    leftPanel.appendChild(playerListArea);

    const rightPanel = document.createElement('div');
    rightPanel.className = 'right-panel';
    mainContainer.appendChild(rightPanel);

    renderPlayerList(playerListArea, gameState, actingPlayerName);
    renderEventLog(rightPanel, gameState, playerMap);
}