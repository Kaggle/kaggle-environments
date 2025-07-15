function renderer({
  environment,
  step,
  parent,
  height = 700, // Default height
  width = 1100, // Default width
}) {
  // --- CSS for the UI ---
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
        .werewolf-parent.night .background {
            background-image: url('https://www.kaggle.com/static/images/games/werewolf/night.png');
        }
        .werewolf-parent.day .background {
            background-image: url('https://www.kaggle.com/static/images/games/werewolf/day.png');
        }
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
        .balloon {
            background-color: #34495e;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .balloon cite {
            font-style: normal;
            font-weight: bold;
            font-size: 0.9em;
            display: block;
            margin-bottom: 5px;
        }
        .msg-entry {
            background-color: rgba(0,0,0,0.2);
            border-left: 3px solid #f39c12;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
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
        .msg-entry cite {
            font-style: normal;
            font-weight: bold;
            display: block;
            font-size: 0.8em;
            color: #bdc3c7;
            margin-bottom: 5px;
        }
    `;

  // --- Helper Functions ---

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
        const statusText = `Status: ${player.status}`;

        li.innerHTML = `
            <img src="${player.thumbnail}" alt="${player.name}" class="avatar">
            <div class="player-info">
                <div class="player-name" title="${player.name}">${player.name}</div>
                <div class="player-role">${roleText}</div>
                <div class="player-status">${statusText}</div>
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
        logEntries.forEach(entry => {
            const li = document.createElement('li');
            let reasoningHtml = entry.reasoning ? `<div class="reasoning-text">"${entry.reasoning}"</div>` : '';

            switch (entry.type) {
                case 'chat':
                    const speaker = playerMap.get(entry.speaker);
                    if (!speaker) return;
                    li.className = 'chat-entry';
                    li.innerHTML = `
                        <img src="${speaker.thumbnail}" alt="${speaker.name}" class="chat-avatar">
                        <div class="balloon">
                            <cite>${speaker.name}</cite>
                            <div class="balloon-text"><quote>${entry.message}</quote></div>
                            ${reasoningHtml}
                        </div>
                    `;
                    break;
                case 'seer_inspection':
                    li.className = 'msg-entry';
                    li.innerHTML = `
                        <cite>Night ${entry.day} (Private)</cite>
                        <div class="msg-text">(As Seer) <strong>${entry.actor}</strong> chose to inspect <strong>${entry.target}</strong>'s role.</div>
                        ${reasoningHtml}
                    `;
                    break;
                case 'seer_inspection_result':
                    li.className = 'msg-entry';
                    li.innerHTML = `
                        <cite>Night ${entry.day} (Private)</cite>
                        <div class="msg-text">(As Seer) <strong>${entry.seer}</strong> saw <strong>${entry.target}</strong>'s role is a <strong>${entry.role}</strong>.</div>
                    `;
                    break;
                case 'doctor_heal_action':
                    li.className = 'msg-entry';
                    li.innerHTML = `
                        <cite>Night ${entry.day} (Private)</cite>
                        <div class="msg-text">(As Doctor) <strong>${entry.actor}</strong> chose to heal <strong>${entry.target}</strong>.</div>
                        ${reasoningHtml}
                    `;
                    break;
                case 'system':
                    li.className = 'msg-entry';
                    li.innerHTML = `<cite>Day ${entry.day}</cite><div class="msg-text">${entry.text}</div>`;
                    break;
                case 'exile':
                    li.className = 'msg-entry game-event';
                    li.innerHTML = `<cite>Day ${entry.day}</cite><div class="msg-text"><strong>${entry.name}</strong> (${entry.role}) was exiled by vote.</div>`;
                    break;
                case 'elimination':
                    li.className = 'msg-entry game-event';
                    li.innerHTML = `<cite>Night ${entry.day}</cite><div class="msg-text"><strong>${entry.name}</strong> was eliminated. Their role was a ${entry.role}.</div>`;
                    break;
                case 'save':
                     li.className = 'msg-entry';
                     li.innerHTML = `<cite>Night ${entry.day} (Doctor Save)</cite><div class="msg-text">Player <strong>${entry.saved_player}</strong> was attacked but saved by a Doctor!</div>`;
                    break;
                case 'vote':
                    li.className = 'msg-entry';
                    li.innerHTML = `
                        <cite>Day ${entry.day} (Vote)</cite>
                        <div class="msg-text"><strong>${entry.name}</strong> votes to eliminate <strong>${entry.target}</strong>.</div>
                        ${reasoningHtml}
                     `;
                     break;
                case 'night_vote':
                    li.className = 'msg-entry';
                    li.innerHTML = `
                        <cite>Night ${entry.day} (Secret Vote)</cite>
                        <div class="msg-text">(As Werewolf) <strong>${entry.name}</strong> votes to eliminate <strong>${entry.target}</strong>.</div>
                        ${reasoningHtml}
                    `;
                    break;
                case 'game_over':
                    li.className = 'msg-entry game-win';
                    li.innerHTML = `
                        <cite>Game Over</cite>
                        <div class="msg-text">
                            <div>The <strong>${entry.winner}</strong> team has won!</div><br>
                            <div><strong>Winning Team:</strong> ${entry.winners.join(', ')}</div>
                            <div><strong>Losing Team:</strong> ${entry.losers.join(', ')}</div>
                        </div>
                    `;
                    break;
            }
            logUl.appendChild(li);
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
    let gameState = { players: [], day: 0, phase: 'GAME_SETUP', gameWinner: null, eventLog: [] };
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

    const roleAndTeamMap = new Map();
    const initialStep = environment.steps[0];
    if (initialStep) {
        for (const agentState of initialStep) {
            const rawObs = agentState.observation?.raw_observation;
            if (rawObs?.new_visible_raw_data) {
                for (const dataEntry of rawObs.new_visible_raw_data) {
                    if (dataEntry.data_type === 'GameStartRoleDataEntry') {
                        const historyEvent = JSON.parse(dataEntry.json_str);
                        const data = historyEvent.data;
                        if (data) roleAndTeamMap.set(data.player_id, { role: data.role, team: data.team });
                    }
                }
            }
        }
    }
    roleAndTeamMap.forEach((info, playerId) => {
        const player = playerMap.get(playerId);
        if (player) { player.role = info.role; player.team = info.team; }
    });

    const processedEvents = new Set();
    let lastPhase = null;
    let lastDay = 0;

    for (let s = 0; s <= step; s++) {
        const stepStateList = environment.steps[s];
        if (!stepStateList) continue;

        const currentObsForStep = stepStateList[0]?.observation?.raw_observation;
        if (currentObsForStep) {
            if (currentObsForStep.day > lastDay) {
                if (currentObsForStep.day > 0) gameState.eventLog.push({ type: 'system', day: currentObsForStep.day, text: `Day ${currentObsForStep.day} has begun.` });
            }
            lastDay = currentObsForStep.day;
            if (lastPhase && lastPhase !== currentObsForStep.phase) gameState.eventLog.push({ type: 'system', day: currentObsForStep.day, text: `Phase: ${currentObsForStep.phase.replace(/_/g, ' ')}` });
            lastPhase = currentObsForStep.phase;
            gameState.day = currentObsForStep.day;
            gameState.phase = currentObsForStep.phase;
            gameState.game_state_phase = currentObsForStep.game_state_phase;
            const alivePlayerIds = new Set(currentObsForStep.alive_players);
            gameState.players.forEach(p => p.is_alive = alivePlayerIds.has(p.name));
        }

        for (const agentState of stepStateList) {
            if (agentState.action) {
                try {
                    const action = typeof agentState.action === 'string' ? JSON.parse(agentState.action) : agentState.action;
                    const actionData = action.kwargs;
                    const actionType = action.action_type;

                    if (actionData && actionType) {
                        const actionKey = `${s}-${actionData.actor_id}-${actionType}`;
                        if (!processedEvents.has(actionKey)) {
                            processedEvents.add(actionKey);
                            const commonData = { day: gameState.day, reasoning: actionData.reasoning };
                            switch (actionType) {
                                case 'ChatAction':
                                    gameState.eventLog.push({ ...commonData, type: 'chat', speaker: actionData.actor_id, message: actionData.message });
                                    break;
                                case 'VoteAction':
                                    gameState.eventLog.push({ ...commonData, type: 'vote', name: actionData.actor_id, target: actionData.target_id });
                                    break;
                                case 'EliminateProposalAction':
                                    gameState.eventLog.push({ ...commonData, type: 'night_vote', name: actionData.actor_id, target: actionData.target_id });
                                    break;
                                case 'HealAction':
                                    gameState.eventLog.push({ ...commonData, type: 'doctor_heal_action', actor: actionData.actor_id, target: actionData.target_id });
                                    break;
                                case 'InspectAction':
                                    gameState.eventLog.push({ ...commonData, type: 'seer_inspection', actor: actionData.actor_id, target: actionData.target_id });
                                    break;
                            }
                        }
                    }
                } catch (e) { /* Fail silently */ }
            }

            if (!agentState.observation?.raw_observation?.new_visible_raw_data) continue;

            for (const dataEntry of agentState.observation.raw_observation.new_visible_raw_data) {
                const eventKey = JSON.stringify(dataEntry);
                if (processedEvents.has(eventKey)) continue;
                processedEvents.add(eventKey);

                const historyEvent = JSON.parse(dataEntry.json_str);
                if (!historyEvent.data) continue;
                const data = historyEvent.data;

                switch (dataEntry.data_type) {
                    case 'DayExileElectedDataEntry':
                        gameState.eventLog.push({ type: 'exile', day: historyEvent.day, name: data.elected_player_id, role: data.elected_player_role_name });
                        const exiledPlayer = playerMap.get(data.elected_player_id);
                        if (exiledPlayer) { exiledPlayer.is_alive = false; exiledPlayer.status = 'Exiled by voting'; }
                        break;
                    case 'WerewolfNightEliminationDataEntry':
                        gameState.eventLog.push({ type: 'elimination', day: historyEvent.day, name: data.eliminated_player_id, role: data.eliminated_player_role_name });
                        const elimPlayer = playerMap.get(data.eliminated_player_id);
                        if (elimPlayer) { elimPlayer.is_alive = false; elimPlayer.status = 'Eliminated by werewolf'; }
                        break;
                    case 'SeerInspectResultDataEntry':
                        gameState.eventLog.push({ type: 'seer_inspection_result', day: historyEvent.day, seer: data.actor_id, target: data.target_id, role: data.role });
                        break;
                    case 'DoctorSaveDataEntry':
                        gameState.eventLog.push({ type: 'save', day: historyEvent.day, saved_player: data.saved_player_id });
                        break;
                    case 'GameEndResultsDataEntry':
                        gameState.gameWinner = data.winner_team;
                        const winners = gameState.players.filter(p => p.team === data.winner_team).map(p => p.name);
                        const losers = gameState.players.filter(p => p.team !== data.winner_team).map(p => p.name);
                        gameState.eventLog.push({ type: 'game_over', day: Infinity, winner: data.winner_team, winners, losers });
                        break;
                }
            }
        }
    }

    gameState.eventLog.sort((a, b) => a.day - b.day);

    const lastStepStateList = environment.steps[step];
    const actingPlayerIndex = lastStepStateList.findIndex(s => s.status === 'ACTIVE');
    const actingPlayerName = actingPlayerIndex !== -1 ? allPlayerNamesList[actingPlayerIndex] : "N/A";

    parent.innerHTML = '';
    Object.assign(parent.style, { width: `${width}px`, height: `${height}px` });
    parent.className = 'werewolf-parent';

    const style = document.createElement('style');
    style.textContent = css;
    parent.appendChild(style);

    const isNight = gameState.game_state_phase === 'NIGHT';
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