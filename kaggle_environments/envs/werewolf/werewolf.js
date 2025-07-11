// werewolf.js
function renderer({
    environment,
    step,
    parent,
    height = 600, // Default height
    width = 800   // Default width
}) {
    // Helper to get player name or a special string if index is out of bounds
    function formatVotes(votesObject) {
        if (!votesObject || Object.keys(votesObject).length === 0) {
            return "No votes cast yet.";
        }
        let readableVotes = "";
        for (const voterName in votesObject) {
            const targetName = votesObject[voterName];
            readableVotes += `  ${voterName} -> ${targetName}\n`;
        }
        return readableVotes.trim();
    }

    // Helper to parse discussion log into a readable string
    function formatDiscussion(discussionLog) {
        if (!discussionLog || discussionLog.length === 0) {
            return "No discussion yet this day.";
        }
        try {
            return discussionLog.map(d => `${d.speaker}: ${d.message}`).join('\n');
        } catch (e) {
            return "Error parsing discussion log.";
        }
    }

    parent.innerHTML = ''; // Clear previous rendering

    const container = document.createElement("div");
    Object.assign(container.style, {
        fontFamily: "Arial, sans-serif",
        padding: "15px",
        backgroundColor: "#f4f4f9",
        border: "1px solid #ccc",
        borderRadius: "8px",
        width: `${width - 32}px`,
        height: `${height - 32}px`,
        overflowY: "auto",
    });

    if (!environment || !environment.steps || environment.steps.length === 0 || step >= environment.steps.length) {
        container.textContent = "Waiting for game data or invalid step...";
        parent.appendChild(container);
        return;
    }
    
    // --- State Reconstruction ---
    // The new observation model is event-based, so we must reconstruct state by processing history.
    let gameState = {
        players: [],
        day: 0,
        phase: 'GAME_SETUP',
        gameWinner: null,
        lastLynched: null,
        lastKilledByWW: null,
        seerInspections: [],
        discussionLog: [],
        dayVotes: {},
        nightVotes: {},
    };

    const processedEvents = new Set();

    const firstObs = environment.steps[0]?.[0]?.observation?.raw_observation;
    if (!firstObs) {
        container.textContent = "Waiting for game data...";
        parent.appendChild(container);
        return;
    }

    const allPlayerNamesList = firstObs.all_player_ids;
    gameState.players = allPlayerNamesList.map(name => ({ name: name, is_alive: true, role: 'Unknown' }));
    const playerMap = new Map(gameState.players.map(p => [p.name, p]));

    for (let s = 0; s <= step; s++) {
        const stepStateList = environment.steps[s];
        if (!stepStateList) continue;

        const currentObsForStep = stepStateList[0]?.observation?.raw_observation;
        if (currentObsForStep) {
            gameState.day = currentObsForStep.day;
            gameState.phase = currentObsForStep.phase;
            // Update alive status based on the latest step's observation
            const alivePlayerIds = new Set(currentObsForStep.alive_players);
            for (const player of gameState.players) {
                player.is_alive = alivePlayerIds.has(player.name);
            }
        }

        for (const agentState of stepStateList) {
            if (!agentState.observation?.raw_observation?.new_visible_raw_data) continue;

            for (const dataEntry of agentState.observation.raw_observation.new_visible_raw_data) {
                const eventKey = JSON.stringify(dataEntry);
                if (processedEvents.has(eventKey)) continue;
                processedEvents.add(eventKey);

                // The json_str in the current Python implementation contains the entire HistoryEntry object.
                // The actual event payload is in the 'data' property of that object.
                const historyEvent = JSON.parse(dataEntry.json_str);
                if (!historyEvent.data) continue; // Safety check for events without a data payload
                const data = historyEvent.data;

                switch (dataEntry.data_type) {
                    case 'GameStartRoleDataEntry':
                        playerMap.get(data.player_id).role = data.role;
                        break;
                    case 'DayExileElectedDataEntry':
                        gameState.lastLynched = { name: data.elected_player_id, role: data.elected_player_role_name };
                        if (playerMap.has(data.elected_player_id)) {
                            playerMap.get(data.elected_player_id).is_alive = false;
                        }
                        gameState.discussionLog = []; // Reset for next day
                        break;
                    case 'WerewolfNightEliminationDataEntry':
                        gameState.lastKilledByWW = { name: data.eliminated_player_id, role: data.eliminated_player_role_name };
                        if (playerMap.has(data.eliminated_player_id)) {
                            playerMap.get(data.eliminated_player_id).is_alive = false;
                        }
                        break;
                    case 'SeerInspectResultDataEntry':
                        gameState.seerInspections.push({ seer: data.actor_id, target: data.target_id, role: data.role, day: gameState.day });
                        break;
                    case 'ChatDataEntry':
                        gameState.discussionLog.push({ speaker: data.speaker_id, message: data.message });
                        break;
                    case 'DayExileVoteDataEntry':
                        if (!gameState.dayVotes[gameState.day]) gameState.dayVotes[gameState.day] = {};
                        gameState.dayVotes[gameState.day][data.actor_id] = data.target_id;
                        break;
                    case 'WerewolfNightVoteDataEntry':
                        if (!gameState.nightVotes[gameState.day]) gameState.nightVotes[gameState.day] = {};
                        gameState.nightVotes[gameState.day][data.actor_id] = data.target_id;
                        break;
                    case 'GameEndResultsDataEntry':
                        gameState.gameWinner = data.winner_team;
                        Object.entries(data.all_players_and_role).forEach(([p_id, p_role]) => {
                            playerMap.get(p_id).role = p_role;
                        });
                        break;
                }
            }
        }
    }

    // --- Actor Info ---
    const lastStepStateList = environment.steps[step];
    const actingPlayerIndex = lastStepStateList.findIndex(s => s.status === 'ACTIVE');
    const actingPlayerName = actingPlayerIndex !== -1 ? allPlayerNamesList[actingPlayerIndex] : "N/A";

    // --- Game Info Section ---
    const gameInfoDiv = document.createElement("div");
    gameInfoDiv.innerHTML = `
        <h2 style="color: #333; border-bottom: 2px solid #666; padding-bottom: 5px;">Werewolf Game Viewer</h2>
        <p><strong>Kaggle Step:</strong> ${step}</p>
        <p><strong>Day:</strong> ${gameState.day}</p>
        <p><strong>Game Phase:</strong> ${gameState.phase.replace(/_/g, ' ')}</p>
        ${actingPlayerName !== "N/A" ? `
            <p><strong>Awaiting Action From:</strong> ${actingPlayerName}</p>
        ` : '<p><strong>Action:</strong> Game is processing...</p>'}
    `;
    container.appendChild(gameInfoDiv);

    // --- Player Status Section ---
    const playerStatusDiv = document.createElement("div");
    playerStatusDiv.innerHTML = `<h3 style="color: #555;">Player Status</h3>`;
    const playerListUl = document.createElement("ul");
    Object.assign(playerListUl.style, { listStyleType: "none", paddingLeft: "0" });

    for (const player of gameState.players) {
        const li = document.createElement("li");
        Object.assign(li.style, { padding: "5px 0", borderBottom: "1px dashed #ddd" });

        let roleDisplay = "Role: Unknown";
        if (gameState.gameWinner) {
            roleDisplay = `Role: ${player.role} (${player.is_alive ? 'Survived' : 'Eliminated'})`;
        } else if (!player.is_alive) {
            roleDisplay = `Role: ${player.role} (Eliminated)`;
        } else {
            roleDisplay = `Role: (Alive)`;
        }

        li.innerHTML = `
            <strong style="color: ${player.is_alive ? 'green' : 'red'};">${player.name}</strong>
            (Status: ${player.is_alive ? 'Alive' : 'Dead'}) - ${roleDisplay}
        `;
        if (player.name === actingPlayerName) {
            li.style.backgroundColor = "#e0e0ff"; // Highlight acting player
        }
        playerListUl.appendChild(li);
    }
    playerStatusDiv.appendChild(playerListUl);
    container.appendChild(playerStatusDiv);

    // --- Events & Details Section ---
    const eventLogDiv = document.createElement("div");
    eventLogDiv.innerHTML = `<h3 style="color: #555;">Events & Details</h3>`;

    if (gameState.lastLynched) {
        eventLogDiv.innerHTML += `<p><strong>Last Lynched:</strong> ${gameState.lastLynched.name} (Role: ${gameState.lastLynched.role})</p>`;
    }
    if (gameState.lastKilledByWW) {
        eventLogDiv.innerHTML += `<p><strong>Last Killed by Werewolf:</strong> ${gameState.lastKilledByWW.name} (Role: ${gameState.lastKilledByWW.role})</p>`;
    }

    // Discussion Log
    const discussionDiv = document.createElement("div");
    discussionDiv.innerHTML = `<h4>Discussion Log (Current Day)</h4>`;
    const discussionPre = document.createElement("pre");
    Object.assign(discussionPre.style, {
        backgroundColor: "#fff", border: "1px solid #eee", padding: "10px",
        maxHeight: "150px", overflowY: "auto", whiteSpace: "pre-wrap"
    });
    discussionPre.textContent = formatDiscussion(gameState.discussionLog);
    discussionDiv.appendChild(discussionPre);
    eventLogDiv.appendChild(discussionDiv);

    // Vote Details
    const voteDetailsDiv = document.createElement("div");
    voteDetailsDiv.innerHTML = `<h4>Vote Details</h4>`;

    const currentDayVotes = gameState.dayVotes[gameState.day] || {};
    const dayVotesPre = document.createElement("pre");
    dayVotesPre.textContent = `Day ${gameState.day} Lynch Votes:\n${formatVotes(currentDayVotes)}`;
    voteDetailsDiv.appendChild(dayVotesPre);

    const currentNightVotes = gameState.nightVotes[gameState.day] || {};
    if (Object.keys(currentNightVotes).length > 0) {
        const nightVotesPre = document.createElement("pre");
        nightVotesPre.textContent = `Night ${gameState.day} Werewolf Votes:\n${formatVotes(currentNightVotes)}`;
        voteDetailsDiv.appendChild(nightVotesPre);
    }

    eventLogDiv.appendChild(voteDetailsDiv);

    // Seer Inspections
    const seerInspectionsForDay = gameState.seerInspections.filter(insp => insp.day === gameState.day);
    if (seerInspectionsForDay.length > 0) {
        const seerInspectionDiv = document.createElement("div");
        seerInspectionDiv.innerHTML = `<h4>Seer Inspections (Night ${gameState.day})</h4>`;
        seerInspectionsForDay.forEach(insp => {
            seerInspectionDiv.innerHTML += `<p>${insp.seer} inspected ${insp.target} and saw role: ${insp.role}</p>`;
        });
        eventLogDiv.appendChild(seerInspectionDiv);
    }

    container.appendChild(eventLogDiv);

    // --- Game Outcome Section ---
    if (gameState.gameWinner) {
        const outcomeDiv = document.createElement("div");
        outcomeDiv.innerHTML = `
            <h2 style="color: #800000; margin-top: 20px;">GAME OVER!</h2>
            <p style="font-size: 1.2em;"><strong>Winner Team:</strong> ${gameState.gameWinner}</p>
        `;
        container.appendChild(outcomeDiv);
    }

    parent.appendChild(container);
}