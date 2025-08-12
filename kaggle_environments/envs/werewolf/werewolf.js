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
            position: relative;
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
        .avatar-container {
            position: relative;
            width: 50px;
            height: 50px;
            margin-right: 15px;
            flex-shrink: 0;
        }
        .player-card .avatar {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            object-fit: cover;
            background-color: #fff;
            transition: box-shadow 0.3s ease;
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
        .player-role,         .player-status {
            font-size: 0.9em;
            color: #bdc3c7;
        }
        .threat-indicator {
            position: absolute;
            top: 50%;
            right: 15px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            transform: translateY(-50%);
            background-color: transparent;
            transition: background-color 0.3s ease;
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
             line-height: 1.5;
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
            line-height: 1.2;
        }
        .moderator-announcement-content.event-day {
            background-color: rgba(236, 240, 241, 0.1);
        }
        .moderator-announcement-content.event-night {
            background-color: rgba(0, 0, 0, 0.25);
        }
        .timestamp {
            font-size: 0.8em;
            color: #bdc3c7;
            margin-left: 10px;
            font-weight: normal;
        }
        .msg-text br {
            display: block; /* makes <br> behave like a block element */
            margin-bottom: 0.5em; /* space below the <br> */
            content: ""; /* required for margin to apply */
        }
        .player-capsule {
            display: inline-flex;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1px 8px 1px 2px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 0 2px;
            vertical-align: middle;
        }
        .capsule-avatar {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            margin-right: 5px;
            object-fit: cover;
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

  function createPlayerCapsule(player) {
    if (!player) return '';
    return `<span class="player-capsule" title="${player.name}">
        <img src="${player.thumbnail}" class="capsule-avatar" alt="${player.name}">
        <span class="capsule-name">${player.name}</span>
    </span>`;
  }

  function replacePlayerIdsWithCapsules(text, playerIds, playerMap) {
      if (!text) return '';
      if (!playerIds || playerIds.length === 0) {
          return text;
      }
      let newText = text;
      // Sort by length descending to match longer names first (e.g. "Player10" before "Player1")
      const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

      sortedPlayerIds.forEach(playerId => {
          const player = playerMap.get(playerId);
          if (player) {
              const capsule = createPlayerCapsule(player);
              // Use a regex to replace whole words only to avoid replacing parts of other words.
              // The \b is a word boundary.
              const regex = new RegExp(`\\b${playerId.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')}\\b`, 'g');
              newText = newText.replace(regex, capsule);
          }
      });
      return newText;
  }

  function replacePlayerIdsWithBold(text, playerIds) {
    if (!text) return '';
    if (!playerIds || playerIds.length === 0) {
        return text;
    }
    let newText = text;
    // Sort by length descending to match longer names first (e.g. "Player10" before "Player1")
    const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

    sortedPlayerIds.forEach(playerId => {
        // Use a regex to replace whole words only to avoid replacing parts of other words.
        const regex = new RegExp(`\\b${playerId.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')}\\b`, 'g');
        newText = newText.replace(regex, `<strong>${playerId}</strong>`);
    });
    return newText;
  }


  function getThreatColor(threatLevel) {
    const value = Math.max(0, Math.min(1, threatLevel));
    // Interpolates from green (hue 120) to red (hue 0)
    const hue = 120 * (1 - value);
    // Use HSL for vibrant colors
    return `hsl(${hue}, 100%, 50%)`;
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

        let roleDisplay = player.role;
        if (player.role === 'Werewolf') {
            roleDisplay = `\uD83D\uDC3A ${player.role}`;
        } else if (player.role === 'Doctor') {
            roleDisplay = `\uD83E\uDE79 ${player.role}`;
        } else if (player.role === 'Seer') {
            roleDisplay = `\uD83D\uDC41\uFE0F ${player.role}`;
        } else if (player.role === 'Villager') {
            roleDisplay = `\uD83D\uDE36 ${player.role}`;
        }

        const roleText = player.role !== 'Unknown' ? `Role: ${roleDisplay}` : 'Role: Unknown';

        li.innerHTML = `
            <div class="avatar-container">
                <img src="${player.thumbnail}" alt="${player.name}" class="avatar">
            </div>
            <div class="player-info">
                <div class="player-name" title="${player.name}">${player.name}</div>
                <div class="player-role">${roleText}</div>
            </div>
            <div class="threat-indicator"></div>
        `;
        playerUl.appendChild(li);
    });

    listContainer.appendChild(playerUl);
    container.appendChild(listContainer);

    // --- Threat Indicator Management ---
    gameState.players.forEach((player, index) => {
        const li = playerUl.children[index];
        const indicator = li.querySelector('.threat-indicator');
        if (!indicator) return;

        if (player.is_alive) {
            const threatLevel = gameState.playerThreatLevels.get(player.name) || 0;
            indicator.style.backgroundColor = getThreatColor(threatLevel);
        } else {
            indicator.style.backgroundColor = 'transparent';
        }
    });
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

            // --- Phase Correction Logic ---
            let phase = (entry.phase || 'Day').toUpperCase();
            const entryType = entry.type;
            const systemText = (entry.text || '').toLowerCase();

            if (entryType === 'exile' || entryType === 'vote' || entryType === 'timeout' || (entryType === 'system' && (systemText.includes('discussion') || systemText.includes('voting for exile')))) {
                phase = 'DAY';
            } else if (
                entryType === 'elimination' || entryType === 'save' || entryType === 'night_vote' ||
                entryType === 'seer_inspection' || entryType === 'seer_inspection_result' ||
                entryType === 'doctor_heal_action' ||
                (entryType === 'system' && (systemText.includes('werewolf vote request') || systemText.includes('doctor save request') || systemText.includes('seer inspect request') || systemText.includes('night has begun')))
            ) {
                phase = 'NIGHT';
            }

            const phaseClass = `event-${phase.toLowerCase()}`;

            let phaseEmoji = phase;
            if (phase === 'DAY') {
                phaseEmoji = '\u2600\uFE0F'; // Sun emoji
            } else if (phase === 'NIGHT') {
                phaseEmoji = '\uD83C\uDF19'; // Crescent moon emoji
            }

            const dayPhaseString = entry.day !== Infinity ? `[D${entry.day} ${phaseEmoji}]` : '';
            const timestampHtml = `<span class="timestamp">${dayPhaseString} ${formatTimestamp(entry.timestamp)}</span>`;

            switch (entry.type) {
                case 'chat':
                    const speaker = playerMap.get(entry.speaker);
                    if (!speaker) return;
                    const messageText = replacePlayerIdsWithBold(entry.message, entry.mentioned_player_ids);
                    li.className = `chat-entry event-day`;
                    li.innerHTML = `
                        <img src="${speaker.thumbnail}" alt="${speaker.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${speaker.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text"><quote>${messageText}</quote></div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'seer_inspection':
                    const seerInspector = playerMap.get(entry.actor_id);
                    if (!seerInspector) return;
                    const seerTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    const seerCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${seerInspector.thumbnail}" alt="${seerInspector.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Secret Inspect by ${seerInspector.name} (Seer) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${seerCap} chose to inspect ${seerTargetCap}'s role.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'seer_inspection_result':
                    const seerResultViewer = playerMap.get(entry.seer);
                    if (!seerResultViewer) return;
                    const seerCap_ = createPlayerCapsule(playerMap.get(entry.seer));
                    const seerResultTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    li.className = `chat-entry ${phaseClass}`;
                    li.innerHTML = `
                        <img src="${seerResultViewer.thumbnail}" alt="${seerResultViewer.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${entry.seer} (Seer) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${seerCap_} saw ${seerResultTargetCap}'s role is a <strong>${entry.role}</strong>.</div>
                            </div>
                        </div>
                    `;
                    break;
                case 'doctor_heal_action':
                    const doctor = playerMap.get(entry.actor_id);
                    if (!doctor) return;
                    const docTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    const docCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${doctor.thumbnail}" alt="${doctor.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Secret Heal by ${doctor.name} (Doctor) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${docCap} chose to heal ${docTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'system':
                    if (entry.text && entry.text.includes('has begun')) return;

                    let systemText = entry.text;
                    // Regex to find python list of strings and replace it with just the comma-separated content
                    const listRegex = /\\\[(.*?)\\\]/g;
                    systemText = systemText.replace(listRegex, (match, listContent) => {
                        // listContent is "'player-1', 'player-2', 'player-3'"
                        return listContent.replace(/'/g, "").replace(/, /g, " "); // Becomes "player-1 player-2 player-3"
                    });

                    const allPlayerIdsForSystem = Array.from(playerMap.keys());
                    const finalSystemText = replacePlayerIdsWithCapsules(systemText, allPlayerIdsForSystem, playerMap);

                    li.className = `moderator-announcement`;
                    li.innerHTML = `
                        <cite>Moderator &#128226; ${timestampHtml}</cite>
                        <div class="moderator-announcement-content ${phaseClass}">
                            <div class="msg-text">${finalSystemText.replace(/\n/g, '<br>')}</div>
                        </div>
                    `;
                    break;
                case 'exile':
                    const exiledPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-day`;
                    li.innerHTML = `<cite>Exile ${timestampHtml}</cite><div class="msg-text">${exiledPlayerCap} (${entry.role}) was exiled by vote.</div>`;
                    break;
                case 'elimination':
                    const elimPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-night`;
                    li.innerHTML = `<cite>Elimination ${timestampHtml}</cite><div class="msg-text">${elimPlayerCap} was eliminated. Their role was a ${entry.role}.</div>`;
                    break;
                case 'save':
                     const savedPlayerCap = createPlayerCapsule(playerMap.get(entry.saved_player));
                     li.className = `msg-entry event-night`;
                     li.innerHTML = `<cite>Doctor Save ${timestampHtml}</cite><div class="msg-text">Player ${savedPlayerCap} was attacked but saved by a Doctor!</div>`;
                    break;
                case 'vote':
                    const voter = playerMap.get(entry.actor_id);
                    if (!voter) return;
                    const voterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    const voteTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    li.className = `chat-entry event-day`;
                    li.innerHTML = `
                        <img src="${voter.thumbnail}" alt="${voter.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${voter.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${voterCap} votes to eliminate ${voteTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                     `;
                     break;
                case 'timeout':
                    const to_voter = playerMap.get(entry.actor_id);
                    if (!to_voter) return;
                    const to_voterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    li.className = `chat-entry event-day`;
                    li.innerHTML = `
                        <img src="${to_voter.thumbnail}" alt="${to_voter.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${to_voter.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${to_voterCap} timed out and abstained from voting.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                     `;
                     break;
                case 'night_vote':
                    const nightVoter = playerMap.get(entry.actor_id);
                    if (!nightVoter) return;
                    const nightVoterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    const nightVoteTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${nightVoter.thumbnail}" alt="${nightVoter.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Secret Vote by ${nightVoter.name} (Werewolf) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${nightVoterCap} votes to eliminate ${nightVoteTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'game_over':
                    const winnersText = entry.winners.map(p => createPlayerCapsule(playerMap.get(p))).join(' ');
                    const losersText = entry.losers.map(p => createPlayerCapsule(playerMap.get(p))).join(' ');
                    li.className = `msg-entry game-win ${phaseClass}`;
                    li.innerHTML = `
                        <cite>Game Over ${timestampHtml}</cite>
                        <div class="msg-text">
                            <div>The <strong>${entry.winner}</strong> team has won!</div><br>
                            <div><strong>Winning Team:</strong> ${winnersText}</div>
                            <div><strong>Losing Team:</strong> ${losersText}</div>
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
    let gameState = {
        players: [],
        day: 0,
        phase: 'GAME_SETUP',
        game_state_phase: 'DAY',
        gameWinner: null,
        eventLog: [],
        playerThreatLevels: new Map()
    };
    const firstObs = environment.steps[0]?.[0]?.observation?.raw_observation;
    if (!firstObs) {
        container.textContent = "Waiting for game data...";
        parent.appendChild(container);
        return;
    }
    const agentConfig = environment.configuration.agents
    const playerThumbnails = firstObs.player_thumbnails || {};
    const allPlayerNamesList = firstObs.all_player_ids;
//    gameState.players = allPlayerNamesList.map(name => ({
//        name: name, is_alive: true, role: 'Unknown', team: 'Unknown', status: 'Alive',
//        thumbnail: playerThumbnails[name] || `https://via.placeholder.com/40/2c3e50/ecf0f1?text=${name.charAt(0)}`
//    }));
    gameState.players = agentConfig.map(p => ({
	name: p.id, is_alive: true, role: p.role, team: 'Unknown', status: 'Alive', thumbnail: p.thumbnail}))
    const playerMap = new Map(gameState.players.map(p => [p.name, p]));

    // Initialize all threat levels to 0 (SAFE)
    gameState.players.forEach(p => gameState.playerThreatLevels.set(p.name, 0));

    // Get initial roles from the start of the game
    const roleAndTeamMap = new Map();
    const moderatorInitialLog = environment.info?.MODERATOR_OBSERVATION?.[0] || [];
    moderatorInitialLog.flat().forEach(dataEntry => {
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

    function threatStringToLevel(threatString) {
        switch(threatString) {
            case 'SAFE': return 0;
            case 'UNEASY': return 0.5;
            case 'IN_DANGER': return 1.0;
            default: return 0; // Default to safe
        }
    }

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
        moderatorLogForStep.flat().forEach(dataEntry => {
             const eventKey = dataEntry.json_str;
             if (processedEvents.has(eventKey)) return;
             processedEvents.add(eventKey);

             const historyEvent = JSON.parse(dataEntry.json_str);
             const data = historyEvent.data;
             const timestamp = historyEvent.created_at;

            // Update threat level whenever an action with that info appears
            if (data && data.actor_id && data.perceived_threat_level) {
                const threatScore = threatStringToLevel(data.perceived_threat_level);
                gameState.playerThreatLevels.set(data.actor_id, threatScore);
            }

            if (!data) {
                if (historyEvent.entry_type === 'vote_action') {
                    const match = historyEvent.description.match(/P(player_\d+)/);
                    if (match) {
                        const actor_id = match[1];
                        gameState.eventLog.push({ type: 'timeout', day: historyEvent.day, phase: historyEvent.phase, actor_id: actor_id, reasoning: "Timed out", timestamp: historyEvent.created_at });
                    }
                }
                return;
            }

             switch (dataEntry.data_type) {
                case 'ChatDataEntry':
                    gameState.eventLog.push({ type: 'chat', day: historyEvent.day, phase: historyEvent.phase, speaker: data.actor_id, message: data.message, reasoning: data.reasoning, timestamp, mentioned_player_ids: data.mentioned_player_ids || [] });
                    break;
                case 'DayExileVoteDataEntry':
                    gameState.eventLog.push({ type: 'vote', day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'WerewolfNightVoteDataEntry':
                    gameState.eventLog.push({ type: 'night_vote', day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'DoctorHealActionDataEntry':
                    gameState.eventLog.push({ type: 'doctor_heal_action', day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'SeerInspectActionDataEntry':
                    gameState.eventLog.push({ type: 'seer_inspection', day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
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
                        gameState.eventLog.push({ type: 'system', day: historyEvent.day, phase: historyEvent.phase, text: historyEvent.description, timestamp, data: data});
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

    renderPlayerList(playerListArea, gameState, playerMap, actingPlayerName);
    renderEventLog(rightPanel, gameState, playerMap);
}