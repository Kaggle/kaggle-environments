// werewolf.js
function renderer({
    environment,
    step,
    parent,
    height = 600, // Default height
    width = 800   // Default width
}) {
    const Role = {
        1: "VILLAGER",
        2: "WEREWOLF",
        3: "DOCTOR",
        4: "SEER",
        0: "Unknown/None" // For default/invalid role values in observation
    };

    const Phase = {
        1: "NIGHT_WEREWOLF_VOTE",
        2: "NIGHT_DOCTOR_SAVE",
        3: "NIGHT_SEER_INSPECT",
        4: "DAY_DISCUSSION",
        5: "DAY_VOTING",
        6: "GAME_OVER"
    };

    // Helper to get player name or a special string if index is out of bounds
    function getPlayerNameOrSpecial(idx, playerNamesList, numActualPlayers, specialValStr = "N/A") {
        if (idx >= 0 && idx < numActualPlayers) {
            return playerNamesList[idx];
        } else if (idx === numActualPlayers) { // Convention for "no one" or "invalid"
            return specialValStr;
        }
        return `Invalid_Index_${idx}`;
    }

    // Helper to get role name or a special string
    function getRoleNameOrSpecial(roleVal, specialValStr = "Unknown/None") {
        if (roleVal === 0) return specialValStr;
        return Role[roleVal] || `Invalid_Role_Value_${roleVal}`;
    }

    // Helper to parse vote details JSON into a readable string
    function parseVoteDetailsReadable(voteJsonStr, playerNamesList, numActualPlayers) {
        if (!voteJsonStr || voteJsonStr === "{}") return "No votes yet or none cast.";
        try {
            const votesDictIdx = JSON.parse(voteJsonStr);
            if (Object.keys(votesDictIdx).length === 0) return "No votes cast.";
            let readableVotes = "";
            for (const voterName in votesDictIdx) {
                const targetIdxInt = votesDictIdx[voterName];
                let targetName = getPlayerNameOrSpecial(targetIdxInt, playerNamesList, numActualPlayers, "Invalid Vote Target/No Vote");
                readableVotes += `  ${voterName} -> ${targetName}\n`;
            }
            return readableVotes.trim() || "No votes recorded.";
        } catch (e) {
            return `Error parsing vote details: '${voteJsonStr}'`;
        }
    }

    parent.innerHTML = ''; // Clear previous rendering

    const container = document.createElement("div");
    container.style.fontFamily = "Arial, sans-serif";
    container.style.padding = "15px";
    container.style.backgroundColor = "#f4f4f9";
    container.style.border = "1px solid #ccc";
    container.style.borderRadius = "8px";
    container.style.width = `${width - 30}px`; // Adjust for padding
    container.style.height = `${height - 30}px`;
    container.style.overflowY = "auto";


    if (!environment || !environment.steps || environment.steps.length === 0 || step >= environment.steps.length) {
        container.textContent = "Waiting for game data or invalid step...";
        parent.appendChild(container);
        return;
    }

    const currentKaggleStepStateList = environment.steps[step];
    if (!currentKaggleStepStateList || !currentKaggleStepStateList[0] || !currentKaggleStepStateList[0].observation || !currentKaggleStepStateList[0].observation.raw_aec_observation) {
        if (step === 0 && environment.steps[0] && environment.steps[0][0] && environment.steps[0][0].status === "ACTIVE") {
             // Initial step, might not have raw_aec_observation fully populated in the way later steps do for player 0
             // or interpreter hasn't run yet for the first agent.
             container.textContent = "Initial game state. Waiting for first action.";
        } else {
            container.textContent = "Error: Observation data is missing for this step.";
        }
        parent.appendChild(container);
        return;
    }
    
    // Use player 0's observation for global state, assuming it's consistent for shared fields.
    // If player 0 is dead, find first alive player. If all dead, use player 0 still.
    let agentWithObs = currentKaggleStepStateList[0];
    if (agentWithObs.observation.raw_aec_observation.alive_players && 
        agentWithObs.observation.raw_aec_observation.alive_players[0] === 0) {
        const firstAliveAgent = currentKaggleStepStateList.find(
            ag => ag.observation.raw_aec_observation.alive_players && 
                  ag.observation.raw_aec_observation.alive_players[currentKaggleStepStateList.indexOf(ag)] === 1
        );
        if (firstAliveAgent) agentWithObs = firstAliveAgent;
    }
    const globalRawObs = agentWithObs.observation.raw_aec_observation;


    const allPlayerNamesList = JSON.parse(globalRawObs.all_player_unique_names);
    const numPlayers = allPlayerNamesList.length;

    // Actor info from the 'info' field (should be consistent across agents for this step)
    const actorInfoSource = currentKaggleStepStateList[0].info || {};
    const actingPlayerKaggleIndex = actorInfoSource.actor_for_this_kaggle_step;
    
    let actingPlayerName = "N/A";
    let actionDescription = "N/A";
    let lastActionFeedback = "N/A";

    if (actingPlayerKaggleIndex !== null && actingPlayerKaggleIndex !== undefined && actingPlayerKaggleIndex < numPlayers) {
        actingPlayerName = allPlayerNamesList[actingPlayerKaggleIndex];
        const actingPlayerStateInfo = currentKaggleStepStateList[actingPlayerKaggleIndex].info || {};
        actionDescription = actingPlayerStateInfo.action_description_for_log || "No action description.";
        lastActionFeedback = actingPlayerStateInfo.last_action_feedback || "No feedback.";
    } else if (step === 0) {
        actionDescription = "Initial game setup.";
    }


    // --- Game Info Section ---
    const gameInfoDiv = document.createElement("div");
    gameInfoDiv.innerHTML = `
        <h2 style="color: #333; border-bottom: 2px solid #666; padding-bottom: 5px;">Werewolf Game Viewer</h2>
        <p><strong>Kaggle Step:</strong> ${step}</p>
        <p><strong>Game Phase:</strong> ${Phase[globalRawObs.phase] || 'Unknown'}</p>
        ${step > 0 && actingPlayerName !== "N/A" ? `
            <p><strong>Agent Acted:</strong> ${actingPlayerName} (Index: ${actingPlayerKaggleIndex})</p>
            <p><strong>Action Processed:</strong> ${actionDescription}</p>
            <p><strong>Feedback to Agent:</strong> ${lastActionFeedback}</p>
        ` : '<p><strong>Action:</strong> Initial State</p>'}
    `;
    container.appendChild(gameInfoDiv);

    // --- Player Status Section ---
    const playerStatusDiv = document.createElement("div");
    playerStatusDiv.innerHTML = `<h3 style="color: #555;">Player Status</h3>`;
    const playerListUl = document.createElement("ul");
    playerListUl.style.listStyleType = "none";
    playerListUl.style.paddingLeft = "0";

    for (let i = 0; i < numPlayers; i++) {
        const li = document.createElement("li");
        li.style.padding = "5px 0";
        li.style.borderBottom = "1px dashed #ddd";

        const playerName = allPlayerNamesList[i];
        const isAlive = globalRawObs.alive_players[i] === 1;
        let roleDisplay = "Role: Unknown";

        if (!isAlive) { // If dead, check if role was revealed
            if (i === globalRawObs.last_lynched && globalRawObs.last_lynched_player_role !== 0) {
                roleDisplay = `Role: ${getRoleNameOrSpecial(globalRawObs.last_lynched_player_role)} (Lynched)`;
            } else if (i === globalRawObs.last_killed_by_werewolf && globalRawObs.last_killed_by_werewolf_role !== 0) {
                roleDisplay = `Role: ${getRoleNameOrSpecial(globalRawObs.last_killed_by_werewolf_role)} (Killed by WW)`;
            } else {
                 roleDisplay = `Role: Unknown (Eliminated)`;
            }
        } else {
             // For a general replay, we don't show roles of alive players unless it's the current agent's own role
             // or a WW seeing other WWs, or Seer seeing their result.
             // This part can be enhanced if rendering for a specific agent's perspective.
             // For now, just "Role: Alive" or if it's the acting agent, their role.
             if (i === actingPlayerKaggleIndex && step > 0 && environment.steps[step-1][i].observation.raw_aec_observation) {
                 const actingPlayerPrevObs = environment.steps[step-1][i].observation.raw_aec_observation;
                 roleDisplay = `Role: ${getRoleNameOrSpecial(actingPlayerPrevObs.role)} (Alive)`;
             } else if (i === 0 && step === 0) { // Initial role for player 0
                 roleDisplay = `Role: ${getRoleNameOrSpecial(globalRawObs.role)} (Alive)`;
             }
             else {
                 roleDisplay = `Role: (Alive)`;
             }
        }
        
        li.innerHTML = `
            <strong style="color: ${isAlive ? 'green' : 'red'};">${playerName}</strong>
            (Status: ${isAlive ? 'Alive' : 'Dead'}) - ${roleDisplay}
        `;
        if (i === actingPlayerKaggleIndex) {
            li.style.backgroundColor = "#e0e0ff"; // Highlight acting player
        }
        playerListUl.appendChild(li);
    }
    playerStatusDiv.appendChild(playerListUl);
    container.appendChild(playerStatusDiv);

    // --- Event Log Section ---
    const eventLogDiv = document.createElement("div");
    eventLogDiv.innerHTML = `<h3 style="color: #555;">Events & Details</h3>`;

    const lastLynchedName = getPlayerNameOrSpecial(globalRawObs.last_lynched, allPlayerNamesList, numPlayers, "No One Lynched");
    const lastLynchedRole = getRoleNameOrSpecial(globalRawObs.last_lynched_player_role);
    eventLogDiv.innerHTML += `<p><strong>Last Lynched:</strong> ${lastLynchedName} (Role: ${lastLynchedRole})</p>`;

    const lastKilledName = getPlayerNameOrSpecial(globalRawObs.last_killed_by_werewolf, allPlayerNamesList, numPlayers, "No One Killed by WW");
    const lastKilledRole = getRoleNameOrSpecial(globalRawObs.last_killed_by_werewolf_role);
    eventLogDiv.innerHTML += `<p><strong>Last Killed by Werewolf:</strong> ${lastKilledName} (Role: ${lastKilledRole})</p>`;
    
    // Discussion Log
    const discussionDiv = document.createElement("div");
    discussionDiv.innerHTML = `<h4>Discussion Log (Current Day)</h4>`;
    const discussionPre = document.createElement("pre");
    discussionPre.style.backgroundColor = "#fff";
    discussionPre.style.border = "1px solid #eee";
    discussionPre.style.padding = "10px";
    discussionPre.style.maxHeight = "150px";
    discussionPre.style.overflowY = "auto";
    try {
        const discussion = JSON.parse(globalRawObs.discussion_log);
        if (discussion.length > 0) {
            discussionPre.textContent = discussion.map(d => `${d.speaker}: ${d.message}`).join('\n');
        } else {
            discussionPre.textContent = "No discussion yet this day.";
        }
    } catch (e) {
        discussionPre.textContent = "Error parsing discussion log.";
    }
    discussionDiv.appendChild(discussionPre);
    eventLogDiv.appendChild(discussionDiv);

    // Vote Details
    const voteDetailsDiv = document.createElement("div");
    voteDetailsDiv.innerHTML = `<h4>Vote Details</h4>`;
    const lastDayVotesPre = document.createElement("pre");
    lastDayVotesPre.textContent = `Last Day Lynch Votes:\n${parseVoteDetailsReadable(globalRawObs.last_day_vote_details, allPlayerNamesList, numPlayers)}`;
    voteDetailsDiv.appendChild(lastDayVotesPre);

    if (Phase[globalRawObs.phase] === "DAY_VOTING") {
        const currentDayVotesPre = document.createElement("pre");
        currentDayVotesPre.textContent = `Current Day Lynch Votes (So Far):\n${parseVoteDetailsReadable(globalRawObs.current_day_vote_details, allPlayerNamesList, numPlayers)}`;
        voteDetailsDiv.appendChild(currentDayVotesPre);
    }
    if (Phase[globalRawObs.phase] === "NIGHT_WEREWOLF_VOTE" && globalRawObs.current_night_werewolf_votes && globalRawObs.current_night_werewolf_votes !== "{}") {
         const currentNightWWVotesPre = document.createElement("pre");
         currentNightWWVotesPre.textContent = `Current Night Werewolf Votes (So Far):\n${parseVoteDetailsReadable(globalRawObs.current_night_werewolf_votes, allPlayerNamesList, numPlayers)}`;
         voteDetailsDiv.appendChild(currentNightWWVotesPre);
    }
    eventLogDiv.appendChild(voteDetailsDiv);
    
    // Seer Inspection (Display if available in global obs - typically only for the seer themselves)
    // For replay, this means if the *last acting agent* was a seer and their obs was captured.
    if (globalRawObs.seer_last_inspection && globalRawObs.seer_last_inspection[0] !== numPlayers) {
        const seerInspectionDiv = document.createElement("div");
        const targetIdx = globalRawObs.seer_last_inspection[0];
        const targetRoleVal = globalRawObs.seer_last_inspection[1];
        const targetName = getPlayerNameOrSpecial(targetIdx, allPlayerNamesList, numPlayers, "No Inspection/Invalid Target");
        const targetRoleName = getRoleNameOrSpecial(targetRoleVal, "Role Not Revealed/Invalid");
        seerInspectionDiv.innerHTML = `<h4>Seer's Last Inspection Result</h4><p>Target: ${targetName}, Role Seen: ${targetRoleName}</p>`;
        eventLogDiv.appendChild(seerInspectionDiv);
    }


    container.appendChild(eventLogDiv);

    // --- Game Outcome Section ---
    if (Phase[globalRawObs.phase] === "GAME_OVER") {
        const outcomeDiv = document.createElement("div");
        const winnerTeam = actorInfoSource.game_winner_team || "Unknown";
        outcomeDiv.innerHTML = `
            <h2 style="color: #800000; margin-top: 20px;">GAME OVER!</h2>
            <p style="font-size: 1.2em;"><strong>Winner Team:</strong> ${winnerTeam}</p>
        `;
        container.appendChild(outcomeDiv);
    }

    parent.appendChild(container);
}
