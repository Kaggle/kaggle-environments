
function renderer(props) {
    const { parent, environment, step, width, height } = props;
    const { h, render } = preact;
    const { useState, useEffect, useMemo, useRef } = preactHooks;
    // Correctly access the styled-components instance for creating components
    const styled = window.styled.default;
    const htm = window.htm;
    const html = htm.bind(h);

    // --- Global Styles (using injectGlobal for styled-components v3) ---
    // Use a flag on the window object to ensure styles are only injected once per page load.
    if (!window.werewolfStylesInjected) {
        // Access injectGlobal from the top-level 'window.styled' object for UMD build
        window.styled.injectGlobal`
            :root {
                --night-bg: #2c3e50;
                --day-bg: #3498db;
                --night-text: #ecf0f1;
                --day-text: #2c3e50;
                --dead-filter: grayscale(100%) brightness(50%);
                --active-border: #f1c40f;
            }
            /* Scope box-sizing to the container to avoid interfering with player UI */
            .werewolf-container, .werewolf-container * {
                box-sizing: border-box;
            }
        `;
        window.werewolfStylesInjected = true;
    }


    // --- Styled Components ---
    const WerewolfParent = styled.div`
        position: relative;
        overflow: hidden;
        width: 100%;
        height: 100%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: var(--night-text);
        background-color: ${props => (props.isNight ? 'var(--night-bg)' : 'var(--day-bg)')};
        transition: background-color 1s ease-in-out;
    `;

    const MainContainer = styled.div`
        position: relative;
        z-index: 1;
        display: flex;
        height: 100%;
        width: 100%;
        background-color: rgba(0,0,0,0.3);
    `;

    const Panel = styled.div`
        background-color: rgba(44, 62, 80, 0.85);
        padding: 15px;
        display: flex;
        flex-direction: column;
        height: 100%;
        h1 {
            margin-top: 0;
            text-align: center;
            font-size: 1.5em;
            border-bottom: 2px solid var(--night-text);
            padding-bottom: 10px;
            flex-shrink: 0;
        }
    `;

    const LeftPanel = styled(Panel)`
        width: 300px;
        flex-shrink: 0;
    `;

    const RightPanel = styled(Panel)`
        flex-grow: 1;
    `;

    const PlayerListContainer = styled.div`
        overflow-y: auto;
        scrollbar-width: thin;
        flex-grow: 1;
        padding-right: 10px;
    `;

    const PlayerCardStyled = styled.div`
        position: relative;
        display: flex;
        align-items: center;
        background-color: rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 5px solid transparent;
        transition: all 0.3s ease;
        opacity: ${props => (props.isAlive ? 1 : 0.6)};
        border-left-color: ${props => (props.isActive ? 'var(--active-border)' : 'transparent')};
        box-shadow: ${props => (props.isActive ? '0 0 15px rgba(241, 196, 15, 0.5)' : 'none')};

        .avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            margin-right: 15px;
            object-fit: cover;
            filter: ${props => (props.isAlive ? 'none' : 'var(--dead-filter)')};
        }
        .player-info {
            flex-grow: 1;
            overflow: hidden;
        }
        .player-name {
            font-weight: bold;
            font-size: 1.1em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .player-role {
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
            background-color: ${props => props.threatColor};
            transition: background-color 0.3s ease;
        }
    `;

    const EventLogContainer = styled.ul`
        list-style: none;
        padding: 0;
        margin: 0;
        flex-grow: 1;
        overflow-y: auto;
        scrollbar-width: thin;
    `;

    const LogEntryStyled = styled.li`
        opacity: 0;
        animation: fadeIn 0.5s forwards;
        margin-bottom: 15px;

        @keyframes fadeIn {
            to {
                opacity: 1;
            }
        }

        cite {
            font-style: normal;
            font-weight: bold;
            display: block;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .timestamp {
            font-size: 0.8em;
            color: #bdc3c7;
            margin-left: 10px;
            font-weight: normal;
        }
    `;

    const ChatEntry = styled(LogEntryStyled)`
        display: flex;
        align-items: flex-start;
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
            max-width: 90%;
            word-wrap: break-word;
            background-color: ${props => props.phase === 'NIGHT' ? 'rgba(0, 0, 0, 0.25)' : 'rgba(236, 240, 241, 0.1)'};
        }
        .reasoning-text {
            font-size: 0.85em;
            color: #bdc3c7;
            font-style: italic;
            margin-top: 5px;
            padding-left: 10px;
            border-left: 2px solid #555;
        }
    `;

    const SystemEntry = styled(LogEntryStyled)`
        .msg-entry {
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid ${props => props.isGameOver ? '#2ecc71' : (props.isGameEvent ? '#e74c3c' : '#f39c12')};
            background-color: ${props => props.phase === 'NIGHT' ? 'rgba(0, 0, 0, 0.25)' : 'rgba(236, 240, 241, 0.1)'};
        }
        .msg-text {
            line-height: 1.5;
        }
    `;

    const PlayerCapsuleStyled = styled.span`
        display: inline-flex;
        align-items: center;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1px 8px 1px 2px;
        font-size: 0.9em;
        font-weight: bold;
        margin: 0 2px;
        vertical-align: middle;
        .capsule-avatar {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            margin-right: 5px;
            object-fit: cover;
        }
    `;


    // --- Helper Functions ---
    const formatTimestamp = (isoString) => {
        if (!isoString) return '';
        try {
            return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        } catch (e) {
            return '';
        }
    };

    const getThreatColor = (threatLevel) => {
        const value = Math.max(0, Math.min(1, threatLevel));
        const hue = 120 * (1 - value); // Interpolates from green (120) to red (0)
        return `hsl(${hue}, 100%, 50%)`;
    };

    const threatStringToLevel = (threatString) => {
        switch(threatString) {
            case 'SAFE': return 0;
            case 'UNEASY': return 0.5;
            case 'IN_DANGER': return 1.0;
            default: return 0;
        }
    };

    // --- Components ---

    const PlayerCapsule = ({ player }) => {
        if (!player) return null;
        return html`
            <${PlayerCapsuleStyled} title=${player.name}>
                <img src=${player.thumbnail} class="capsule-avatar" alt=${player.name} />
                <span>${player.name}</span>
            <//>
        `;
    };

    const TextWithCapsules = ({ text, playerMap }) => {
        if (!text) return '';
        const allPlayerIds = Array.from(playerMap.keys());
        if (allPlayerIds.length === 0) {
             return text.split('\n').map((line, i, arr) => i === arr.length - 1 ? line : html`${line}<br />`);
        }

        const sortedPlayerIds = [...allPlayerIds].sort((a, b) => b.length - a.length);
        const regex = new RegExp(`\b(${sortedPlayerIds.map(id => id.replace(/[-\/\\^$*+?.()|[\\]{}]/g, '\\$&')).join('|')})\b`, 'g');
        const parts = text.split(regex);

        return html`
            <span>
                ${parts.map(part => {
                    const player = playerMap.get(part);
                    if (player) {
                        return html`<${PlayerCapsule} player=${player} />`;
                    }
                    return part.split('\n').map((line, i, arr) =>
                        i === arr.length - 1 ? line : html`${line}<br />`
                    );
                })}
            </span>
        `;
    };


    const PlayerCard = ({ player, isActive }) => {
        const roleText = player.role !== 'Unknown' ? `Role: ${player.role}` : 'Role: Unknown';
        return html`
            <${PlayerCardStyled} isAlive=${player.is_alive} isActive=${isActive} threatColor=${getThreatColor(player.threatLevel || 0)}>
                <img src=${player.thumbnail} alt=${player.name} class="avatar" />
                <div class="player-info">
                    <div class="player-name" title=${player.name}>${player.name}</div>
                    <div class="player-role">${roleText}</div>
                </div>
                <div class="threat-indicator"></div>
            <//>
        `;
    };

    const PlayerList = ({ players, actingPlayerName }) => {
        return html`
            <h1>Players</h1>
            <${PlayerListContainer}>
                ${players.map(p => html`<${PlayerCard} key=${p.name} player=${p} isActive=${p.name === actingPlayerName} />`)}
            <//>
        `;
    };

    const LogEntry = ({ entry, playerMap }) => {
        let phase = (entry.phase || 'Day').toUpperCase();
        const entryType = entry.type;
        const systemText = (entry.text || '').toLowerCase();

        if (entryType === 'exile' || entryType === 'vote' || (entryType === 'system' && (systemText.includes('discussion') || systemText.includes('voting for exile')))) {
            phase = 'DAY';
        } else if (
            entryType === 'elimination' || entryType === 'save' || entryType === 'night_vote' ||
            entryType === 'seer_inspection' || entryType === 'seer_inspection_result' ||
            entryType === 'doctor_heal_action' ||
            (entryType === 'system' && (systemText.includes('werewolf vote request') || systemText.includes('doctor save request') || systemText.includes('seer inspect request') || systemText.includes('night has begun')))
        ) {
            phase = 'NIGHT';
        }

        const dayPhaseString = entry.day !== Infinity ? `[D${entry.day} ${phase}]` : '';
        const timestampHtml = html`<span class="timestamp">${dayPhaseString} ${formatTimestamp(entry.timestamp)}</span>`;
        const reasoningHtml = entry.reasoning ? html`<div class="reasoning-text">"${entry.reasoning}"</div>` : null;

        switch (entry.type) {
            case 'chat':
                const speaker = playerMap.get(entry.speaker);
                if (!speaker) return null;
                return html`
                    <${ChatEntry} phase=${phase}>
                        <img src=${speaker.thumbnail} alt=${speaker.name} class="chat-avatar" />
                        <div class="message-content">
                            <cite>${speaker.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <span><${TextWithCapsules} text=${entry.message} playerMap=${playerMap} /></span>
                                ${reasoningHtml}
                            </div>
                        </div>
                    <//>
                `;
            case 'vote':
            case 'night_vote':
            case 'seer_inspection':
            case 'doctor_heal_action':
                 const actor = playerMap.get(entry.actor_id);
                 const target = playerMap.get(entry.target);
                 if (!actor) return null;
                 let title = '';
                 let text = '';
                 switch(entry.type) {
                    case 'vote':
                        title = `${actor.name}`;
                        text = html`Votes to eliminate <${PlayerCapsule} player=${target} />.`;
                        break;
                    case 'night_vote':
                        title = `Secret Vote by ${actor.name} (Werewolf)`;
                        text = html`<${PlayerCapsule} player=${actor} /> votes to eliminate <${PlayerCapsule} player=${target} />.`;
                        break;
                    case 'seer_inspection':
                         title = `Secret Inspect by ${actor.name} (Seer)`;
                         text = html`<${PlayerCapsule} player=${actor} /> chose to inspect <${PlayerCapsule} player=${target} />'s role.`;
                         break;
                    case 'doctor_heal_action':
                         title = `Secret Heal by ${actor.name} (Doctor)`;
                         text = html`<${PlayerCapsule} player=${actor} /> chose to heal <${PlayerCapsule} player=${target} />.`;
                         break;
                 }
                 return html`
                    <${ChatEntry} phase=${phase}>
                        <img src=${actor.thumbnail} alt=${actor.name} class="chat-avatar" />
                        <div class="message-content">
                            <cite>${title} ${timestampHtml}</cite>
                            <div class="balloon">
                                ${text}
                                ${reasoningHtml}
                            </div>
                        </div>
                    <//>
                 `;
            case 'seer_inspection_result':
                const seer = playerMap.get(entry.seer);
                if (!seer) return null;
                return html`
                    <${ChatEntry} phase=${phase}>
                        <img src=${seer.thumbnail} alt=${seer.name} class="chat-avatar" />
                        <div class="message-content">
                            <cite>${seer.name} (Seer) ${timestampHtml}</cite>
                            <div class="balloon">
                                <${PlayerCapsule} player=${seer} /> saw <${PlayerCapsule} player=${playerMap.get(entry.target)} />'s role is a <strong>${entry.role}</strong>.
                            </div>
                        </div>
                    <//>
                `;
            default: // System messages, eliminations, etc.
                let citeText = 'Moderator 📣';
                let msgText;
                switch(entry.type) {
                    case 'system':
                        if (entry.text && entry.text.includes('has begun')) return null;
                        msgText = html`<${TextWithCapsules} text=${entry.text} playerMap=${playerMap} />`;
                        break;
                    case 'exile':
                        citeText = 'Exile';
                        msgText = html`<${PlayerCapsule} player=${playerMap.get(entry.name)} /> (${entry.role}) was exiled by vote.`;
                        break;
                    case 'elimination':
                        citeText = 'Elimination';
                        msgText = html`<${PlayerCapsule} player=${playerMap.get(entry.name)} /> was eliminated. Their role was a ${entry.role}.`;
                        break;
                    case 'save':
                        citeText = 'Doctor Save';
                        msgText = html`Player <${PlayerCapsule} player=${playerMap.get(entry.saved_player)} /> was attacked but saved by a Doctor!`;
                        break;
                    case 'game_over':
                        citeText = 'Game Over';
                        const winners = entry.winners.map(p => playerMap.get(p));
                        const losers = entry.losers.map(p => playerMap.get(p));
                        msgText = html`
                            <div>The <strong>${entry.winner}</strong> team has won!</div><br/>
                            <div><strong>Winning Team:</strong> ${winners.map(p => html`<${PlayerCapsule} player=${p} />`)}</div>
                            <div><strong>Losing Team:</strong> ${losers.map(p => html`<${PlayerCapsule} player=${p} />`)}</div>
                        `;
                        break;
                    default:
                        return null;
                }
                return html`
                    <${SystemEntry} phase=${phase} isGameOver=${entry.type === 'game_over'} isGameEvent=${entry.type === 'exile' || entry.type === 'elimination'}>
                        <div class="msg-entry">
                            <cite>${citeText} ${timestampHtml}</cite>
                            <div class="msg-text">${msgText}</div>
                        </div>
                    <//>
                `;
        }
    };

    const EventLog = ({ eventLog, playerMap }) => {
        const logEl = useRef(null);
        useEffect(() => {
            if (logEl.current) {
                logEl.current.scrollTop = logEl.current.scrollHeight;
            }
        });

        return html`
            <h1>Event Log</h1>
            <${EventLogContainer} ref=${logEl}>
                ${eventLog.length === 0
                    ? html`<${SystemEntry}><div class="msg-entry"><cite>System</cite><div>The game is about to begin...</div></div><//>`
                    : eventLog.map((entry, i) => html`<${LogEntry} key=${i} entry=${entry} playerMap=${playerMap} />`)}
            <//>
        `;
    };

    const App = ({ environment, step, width, height }) => {
        const gameState = useMemo(() => {
            if (!environment || !environment.steps || environment.steps.length === 0 || !environment.id) {
                return null;
            }

            // --- State Reconstruction ---
            const firstObs = environment.steps[0]?.[0]?.observation?.raw_observation;
            if (!firstObs) return null;

            const playerThumbnails = firstObs.player_thumbnails || {};
            const allPlayerNamesList = firstObs.all_player_ids;
            const initialPlayers = allPlayerNamesList.map(name => ({
                name: name, is_alive: true, role: 'Unknown', team: 'Unknown', status: 'Alive',
                thumbnail: playerThumbnails[name] || `https://via.placeholder.com/50/2c3e50/ecf0f1?text=${name.charAt(0)}`,
                threatLevel: 0
            }));
            const playerMap = new Map(initialPlayers.map(p => [p.name, p]));

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
            let eventLog = [];
            let currentPhase = 'DAY';
            let currentDay = 0;

            for (let s = 0; s <= step; s++) {
                const stepStateList = environment.steps[s];
                if (!stepStateList) continue;

                const currentObsForStep = stepStateList[0]?.observation?.raw_observation;
                if (currentObsForStep) {
                    if (currentObsForStep.day > currentDay && currentObsForStep.day > 0) {
                         eventLog.push({ type: 'system', day: currentObsForStep.day, phase: 'DAY', text: `Day ${currentObsForStep.day} has begun.` });
                    }
                    currentDay = currentObsForStep.day;
                    currentPhase = currentObsForStep.game_state_phase;
                }

                const moderatorLogForStep = environment.info?.MODERATOR_OBSERVATION?.[s] || [];
                moderatorLogForStep.forEach(dataEntry => {
                    const eventKey = dataEntry.json_str;
                    if (processedEvents.has(eventKey)) return;
                    processedEvents.add(eventKey);

                    const historyEvent = JSON.parse(dataEntry.json_str);
                    if (!historyEvent.data) return;
                    const data = historyEvent.data;
                    const timestamp = historyEvent.created_at;

                    if (data.actor_id && data.perceived_threat_level) {
                        const player = playerMap.get(data.actor_id);
                        if(player) player.threatLevel = threatStringToLevel(data.perceived_threat_level);
                    }

                    let logItem = { day: historyEvent.day, phase: historyEvent.phase, timestamp, ...data };
                    switch (dataEntry.data_type) {
                        case 'ChatDataEntry':
                            logItem.type = 'chat';
                            logItem.speaker = data.actor_id;
                            logItem.mentioned_player_ids = data.mentioned_player_ids || [];
                            break;
                        case 'DayExileVoteDataEntry': logItem.type = 'vote'; logItem.target = data.target_id; break;
                        case 'WerewolfNightVoteDataEntry': logItem.type = 'night_vote'; logItem.target = data.target_id; break;
                        case 'DoctorHealActionDataEntry': logItem.type = 'doctor_heal_action'; logItem.target = data.target_id; break;
                        case 'SeerInspectActionDataEntry': logItem.type = 'seer_inspection'; logItem.target = data.target_id; break;
                        case 'DayExileElectedDataEntry': logItem.type = 'exile'; logItem.name = data.elected_player_id; logItem.role = data.elected_player_role_name; break;
                        case 'WerewolfNightEliminationDataEntry': logItem.type = 'elimination'; logItem.name = data.eliminated_player_id; logItem.role = data.eliminated_player_role_name; break;
                        case 'SeerInspectResultDataEntry': logItem.type = 'seer_inspection_result'; logItem.seer = data.actor_id; logItem.target = data.target_id; break;
                        case 'DoctorSaveDataEntry': logItem.type = 'save'; logItem.saved_player = data.saved_player_id; break;
                        case 'GameEndResultsDataEntry':
                            logItem.type = 'game_over';
                            logItem.day = Infinity;
                            logItem.winner = data.winner_team;
                            logItem.winners = Array.from(playerMap.values()).filter(p => p.team === data.winner_team).map(p => p.name);
                            logItem.losers = Array.from(playerMap.values()).filter(p => p.team !== data.winner_team).map(p => p.name);
                            break;
                        default:
                            if (historyEvent.entry_type === "moderator_announcement") {
                                logItem.type = 'system';
                                logItem.text = historyEvent.description;
                            } else {
                                logItem.type = 'unknown';
                            }
                            break;
                    }
                    eventLog.push(logItem);
                });
            }

            const finalPlayers = new Map(playerMap);
            finalPlayers.forEach(p => { p.is_alive = true; p.status = 'Alive'; });
            eventLog.forEach(entry => {
                if (entry.type === 'exile' || entry.type === 'elimination') {
                    const player = finalPlayers.get(entry.name);
                    if (player) {
                        player.is_alive = false;
                        player.status = entry.type === 'exile' ? 'Exiled' : 'Eliminated';
                    }
                }
            });

            eventLog.sort((a, b) => {
                if (a.day !== b.day) return a.day - b.day;
                const phaseOrder = { 'DAY': 1, 'NIGHT': 2, 'GAME_OVER': 3 };
                const aOrder = phaseOrder[a.phase?.toUpperCase()] || 99;
                const bOrder = phaseOrder[b.phase?.toUpperCase()] || 99;
                if (aOrder !== bOrder) return aOrder - bOrder;
                if (a.timestamp && b.timestamp) return new Date(a.timestamp) - new Date(b.timestamp);
                return 0;
            });

            const lastStepStateList = environment.steps[step];
            const actingPlayerIndex = lastStepStateList.findIndex(s => s.status === 'ACTIVE');
            const actingPlayerName = actingPlayerIndex !== -1 ? allPlayerNamesList[actingPlayerIndex] : null;

            return {
                players: Array.from(finalPlayers.values()),
                playerMap: finalPlayers,
                eventLog,
                isNight: currentPhase === 'Night',
                actingPlayerName,
            };
        }, [environment.id, step]);

        if (!gameState) {
            return html`<div>Loading or waiting for game data...</div>`;
        }

        return html`
            <${WerewolfParent} className="werewolf-container" isNight=${gameState.isNight} style=${{ width: `${width}px`, height: `${height}px` }}>
                <${MainContainer}>
                    <${LeftPanel}>
                        <${PlayerList} players=${gameState.players} actingPlayerName=${gameState.actingPlayerName} />
                    <//>
                    <${RightPanel}>
                        <${EventLog} eventLog=${gameState.eventLog} playerMap=${gameState.playerMap} />
                    <//>
                <//>
            <//>
        `;
    };

    render(html`<${App} key=${environment.id} ...${props} />`, parent);
}
