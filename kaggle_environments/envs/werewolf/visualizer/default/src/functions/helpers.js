// --- Helper Functions ---
  export function formatTimestamp(isoString) {
    if (!isoString) return '';
    try {
      return new Date(isoString).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
    } catch (e) {
      return '';
    }
  }

  /**
   * Creates a memoized function to replace player IDs with HTML capsules.
   * This function pre-computes and caches sorted player data for efficiency.
   * @param {Map<string, object>} playerMap - A map from player ID to player object.
   * @returns {function(string): string} A function that takes text and returns it with player IDs replaced.
   */
  export function createPlayerIdReplacer(playerMap) {
    // Cache for already processed text strings (memoization)
    const textCache = new Map();

    // --- Pre-computation Cache ---
    const sortedPlayerReplacements = [...playerMap.keys()]
      .sort((a, b) => b.length - a.length) // Sort by length to match longest names first
      .map((playerId) => {
        const player = playerMap.get(playerId);
        if (!player) return null;

        return {
          capsule: createPlayerCapsule(player),
          // IMPROVEMENT: This new regex correctly handles both internal periods in names (e.g., 'gemini-1.5-pro')
          // and sentence-ending periods (e.g., '... says Kai.').
          // Breakdown:
          // 1. (^|[^\w.-])       - The prefix boundary must not be a name character. This is unchanged.
          // 2. (PLAYER_ID)       - The player's name.
          // 3. (\.?)             - Optionally captures a single trailing period.
          // 4. (?![-\w])          - A negative lookahead asserts that the name is not followed by another name character (a-z, 0-9, _, -).
          //                        This is the key part that allows a trailing period to be treated as a boundary.
          regex: new RegExp(`(^|[^\\w.-])(${playerId.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')})(\\.?)(?![\\w-])`, 'g'),
        };
      })
      .filter(Boolean);

    return function (text) {
      if (!text) return '';
      if (textCache.has(text)) {
        return textCache.get(text);
      }

      let newText = text;
      for (const replacement of sortedPlayerReplacements) {
        // The replacement string now uses $3 to append the optionally captured period after the capsule.
        newText = newText.replace(replacement.regex, `$1${replacement.capsule}$3`);
      }

      textCache.set(text, newText);
      return newText;
    };
  }

  export function createPlayerCapsule(player) {
    if (!player) return '';
    let display_name_elem =
      player.display_name && player.name !== player.display_name
        ? `<span class="capsule-display-name">${player.display_name}</span>`
        : '';
    return `<span class="player-capsule" title="${player.name}">
        <img src="${player.thumbnail}" class="capsule-avatar" alt="${player.name}">
        <span class="capsule-name">${player.name}</span>${display_name_elem}
    </span>`;
  }

  export function replacePlayerIdsWithCapsules(text, playerIds, playerMap) {
    if (!text) return '';
    if (!playerIds || playerIds.length === 0) {
      return text;
    }
    let newText = text;
    const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

    sortedPlayerIds.forEach((playerId) => {
      const player = playerMap.get(playerId);
      if (player) {
        const capsule = createPlayerCapsule(player);
        const escapedPlayerId = playerId.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');

        // Using the same improved regex as in the factory function.
        const regex = new RegExp(`(^|[^\\w.-])(${escapedPlayerId})(\\.?)(?![\\w-])`, 'g');

        // The replacement correctly places the captured prefix ($1) and optional period ($3) around the capsule.
        newText = newText.replace(regex, `$1${capsule}$3`);
      }
    });
    return newText;
  }

  export function replacePlayerIdsWithBold(text, playerIds) {
    if (!text) return '';
    if (!playerIds || playerIds.length === 0) {
      return text;
    }
    let newText = text;
    const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

    sortedPlayerIds.forEach((playerId) => {
      const regex = new RegExp(`\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\b`, 'g');
      newText = newText.replace(regex, `<strong>${playerId}</strong>`);
    });
    return newText;
  }

  export function getThreatColor(threatLevel) {
    const value = Math.max(0, Math.min(1, threatLevel));
    const hue = 120 * (1 - value);
    return `hsl(${hue}, 100%, 50%)`;
  }

  export function updatePlayerList(container, gameState, actingPlayerName) {
    // Get or create header
    let header = container.querySelector('h1');
    if (!header) {
      header = document.createElement('h1');
      // Create a span for the title to sit next to the button
      const titleSpan = document.createElement('span');
      titleSpan.textContent = 'Players';
      header.appendChild(titleSpan);

      // Create the reset button
      const resetButton = document.createElement('button');
      resetButton.id = 'reset-view-btn';
      resetButton.className = 'reset-view-btn';
      resetButton.title = 'Reset Camera View';
      resetButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 2v6h6"/><path d="M21 12A9 9 0 0 0 6 5.3L3 8"/><path d="M21 22v-6h-6"/><path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"/></svg>`;

      header.appendChild(resetButton);
      container.appendChild(header);

      // Add the click listener only once, when the button is created
      resetButton.onclick = () => {
        if (threeState && threeState.demo) {
          threeState.demo.resetCameraView();
        }
      };
    }

    // Get or create list container
    let listContainer = container.querySelector('#player-list-container');
    if (!listContainer) {
      listContainer = document.createElement('div');
      listContainer.id = 'player-list-container';
      container.appendChild(listContainer);
    }

    // Get or create player list
    let playerUl = listContainer.querySelector('#player-list');
    if (!playerUl) {
      playerUl = document.createElement('ul');
      playerUl.id = 'player-list';
      listContainer.appendChild(playerUl);
    }

    // Update player cards
    gameState.players.forEach((player, index) => {
      let li = playerUl.children[index];
      if (!li) {
        li = document.createElement('li');
        playerUl.appendChild(li);
      }

      // Add the onclick handler of player's first person perspective
      // This will call the focus function on the Three.js demo instance
      li.onclick = () => {
        if (threeState && threeState.demo) {
          // Get the current widths of the UI panels
          const leftPanel = parent.querySelector('.left-panel');
          const rightPanel = parent.querySelector('.right-panel');
          const leftPanelWidth = leftPanel ? leftPanel.offsetWidth : 0;
          const rightPanelWidth = rightPanel ? rightPanel.offsetWidth : 0;

          // Pass the panel widths to the focus function
          threeState.demo.focusOnPlayer(player.name, leftPanelWidth, rightPanelWidth);
        }
      };

      // Update player card classes
      li.className = 'player-card';
      if (!player.is_alive) li.classList.add('dead');
      if (player.name === actingPlayerName) li.classList.add('active');

      let roleDisplay = player.role;
      if (player.role === 'Werewolf') {
        roleDisplay = `&#x1F43A; ${player.role}`;
      } else if (player.role === 'Doctor') {
        roleDisplay = `&#x1FA7A; ${player.role}`;
      } else if (player.role === 'Seer') {
        roleDisplay = `&#x1F52E; ${player.role}`;
      } else if (player.role === 'Villager') {
        roleDisplay = `&#x1F9D1; ${player.role}`;
      }

      const roleText = player.role !== 'Unknown' ? `Role: ${roleDisplay}` : 'Role: Unknown';

      // Update content
      let player_name_element = `<div class="player-name" title="${player.name}">${player.name}</div>`;
      if (player.display_name && player.display_name !== player.name) {
        player_name_element = `<div class="player-name" title="${player.name}">
                ${player.name}<span class="display-name">${player.display_name}</span>
            </div>`;
      }

      li.innerHTML = `
            <div class="avatar-container">
                <img src="${player.thumbnail}" alt="${player.name}" class="avatar">
            </div>
            <div class="player-info">
                ${player_name_element}
                <div class="player-role">${roleText}</div>
            </div>
            <div class="threat-indicator"></div>
        `;

      // Update threat indicator
      const indicator = li.querySelector('.threat-indicator');
      if (indicator && player.is_alive) {
        const threatLevel = gameState.playerThreatLevels.get(player.name) || 0;
        indicator.style.backgroundColor = getThreatColor(threatLevel);
      } else if (indicator) {
        indicator.style.backgroundColor = 'transparent';
      }
    });

    // Remove excess player cards
    while (playerUl.children.length > gameState.players.length) {
      playerUl.removeChild(playerUl.lastChild);
    }
  }

  export function updateEventLog(container, gameState, playerMap) {
    const audioState = window.kaggleWerewolf;
    const audioToggleDisabled = !audioState.hasAudioTracks;
    const audioToggleEnabled = audioState.isAudioEnabled && !audioToggleDisabled;
    const audioToggleTitle = audioToggleDisabled ? 'Audio Not Available' : 'Toggle Audio';
    const audioToggleIcon = audioToggleEnabled ? '&#x1F50A;' : '&#x1F507;'; // Speaker vs Muted
    const audioToggleClasses = `audio-toggle-btn ${audioToggleDisabled ? 'disabled' : ''} ${audioToggleEnabled ? 'enabled' : ''}`;

    if (!container.querySelector('h1')) {
      container.innerHTML = `
            <h1>
                <span>Events</span>
                <button id="reset-view-btn" class="reset-view-btn" title="Reset Camera View" style="margin-left: auto;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 2v6h6"/><path d="M21 12A9 9 0 0 0 6 5.3L3 8"/><path d="M21 22v-6h-6"/><path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"/></svg>
                </button>
                <div id="header-controls" style="display: flex; align-items: center; gap: 15px;">
                    <button id="global-reasoning-toggle" title="Toggle All Reasoning">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                    </button>
                    <div id="speed-control-container" style="display: flex; align-items: center; gap: 5px;">
                        <input type="range" id="playback-speed" min="0.5" max="2.5" step="0.1" value="${audioState.playbackRate}" style="width: 70px;">
                        <span id="speed-label" style="font-size: 12px; min-width: 30px;">${audioState.playbackRate.toFixed(1)}x</span>
                    </div>
                    <button id="global-audio-toggle" title="${audioToggleTitle}" class="${audioToggleClasses}">
                        ${audioToggleIcon}
                    </button>
                </div>
            </h1>
        `;
      // NEW: Add the collapse functionality to the header
      const header = container.querySelector('h1');
      if (header) {
        header.onclick = () => {
          container.classList.toggle('collapsed');
        };
      }
      // Initially collapse the panel
      container.classList.add('collapsed');
    }

    // Remove the old log if it exists, to rebuild it
    const oldLogUl = container.querySelector('#chat-log');
    if (oldLogUl) {
      oldLogUl.remove();
    }

    const resetButton = container.querySelector('#reset-view-btn');
    if (resetButton) {
      resetButton.onclick = () => {
        if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
          window.werewolfThreeJs.demo.resetCameraView();
        }
      };
    }

    const logUl = document.createElement('ul');
    logUl.id = 'chat-log';

    const logEntries = gameState.eventLog;

    if (logEntries.length === 0) {
      const li = document.createElement('li');
      li.className = 'msg-entry';
      li.innerHTML = `<cite>System</cite><div>The game is about to begin...</div>`;
      logUl.appendChild(li);
    } else {
      logEntries.forEach((entry, entryIndex) => {
        const li = document.createElement('li');
        li.dataset.allEventsIndex = entry.allEventsIndex;
        let reasoningHtml = '';
        let reasoningToggleHtml = '';
        if (entry.reasoning) {
          const reasoningId = `reasoning-${window.werewolfGamePlayer.reasoningCounter++}`;
          reasoningHtml = `<div class="reasoning-text" id="${reasoningId}">"${entry.reasoning}"</div>`;
          reasoningToggleHtml = `<span class="reasoning-toggle" title="Show/Hide Reasoning" onclick="event.stopPropagation(); document.getElementById('${reasoningId}').classList.toggle('visible')">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-eye"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                </span>`;
        }

        let phase = (entry.phase || 'Day').toUpperCase();
        const entryType = entry.type;
        const systemText = (entry.text || '').toLowerCase();

        const phaseClass = `event-${phase.toLowerCase()}`;

        let phaseEmoji = phase;
        if (phase === 'DAY') {
          phaseEmoji = '&#x2600;&#xFE0F;';
        } else if (phase === 'NIGHT') {
          phaseEmoji = '&#x1F319;';
        }

        const dayPhaseString = entry.day !== Infinity ? `[${phaseEmoji} ${entry.day}]` : '';
        const timestampHtml = `<span class="timestamp">${dayPhaseString} ${formatTimestamp(entry.timestamp)}</span>`;

        switch (entry.type) {
          case 'chat':
            const speaker = playerMap.get(entry.speaker);
            if (!speaker) return;
            const messageText = window.werewolfGamePlayer.playerIdReplacer(entry.message);
            li.className = `chat-entry event-day`;
            li.innerHTML = `
                        <img src="${speaker.thumbnail}" alt="${speaker.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>
                                <span> <span>${speaker.name}</span>
                                    ${speaker.display_name && speaker.name !== speaker.display_name ? `<span class="display-name">${speaker.display_name}</span>` : ''}
                                </span> ${reasoningToggleHtml}
                                ${timestampHtml}
                            </cite>
                            <div class="balloon">
                                <div class="balloon-text">
                                    <quote>${messageText}</quote>
                                </div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
            const balloon = li.querySelector('.balloon');
            if (balloon) {
              balloon.onclick = (e) => {
                e.stopPropagation();
                // This will either play (if audio is enabled)
                // or queue the request (if audio is disabled)
                speak(entry.allEventsIndex);
              };
            }
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
                            <cite>Seer Secret Inspect ${reasoningToggleHtml} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${seerCap} chose to inspect ${seerTargetCap}'s role.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
            const seer_balloon = li.querySelector('.balloon');
            if (seer_balloon) {
              seer_balloon.onclick = (e) => {
                e.stopPropagation();
                // This will either play (if audio is enabled)
                // or queue the request (if audio is disabled)
                speak(entry.allEventsIndex);
              };
            }
            break;
          case 'seer_inspection_result':
            const seerResultViewer = playerMap.get(entry.seer);
            if (!seerResultViewer) return;
            const seerCap_ = createPlayerCapsule(playerMap.get(entry.seer));
            const seerResultTargetCap = createPlayerCapsule(playerMap.get(entry.target));
            const resultString = entry.role
              ? `role is a <strong>${entry.role}</strong>`
              : `team is <strong>${entry.team}</strong>`;

            li.className = `chat-entry ${phaseClass}`;
            li.innerHTML = `
                        <img src="${seerResultViewer.thumbnail}" alt="${seerResultViewer.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Seer Inspect Result ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${seerCap_} saw ${seerResultTargetCap}'s ${resultString}.</div>
                            </div>
                        </div>
                    `;
            const seer_balloon_ = li.querySelector('.balloon');
            if (seer_balloon_) {
              seer_balloon_.onclick = (e) => {
                e.stopPropagation();
                // This will either play (if audio is enabled)
                // or queue the request (if audio is disabled)
                speak(entry.allEventsIndex);
              };
            }
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
                            <cite>Doctor Secret Heal ${reasoningToggleHtml} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${docCap} chose to heal ${docTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
            const dr_balloon = li.querySelector('.balloon');
            if (dr_balloon) {
              dr_balloon.onclick = (e) => {
                e.stopPropagation();
                // This will either play (if audio is enabled)
                // or queue the request (if audio is disabled)
                speak(entry.allEventsIndex);
              };
            }
            break;
          case 'system':
            if (entry.text && entry.text.includes('has begun')) return;

            let systemText = entry.text;

            // This enhanced regex captures the list content (group 1) and any optional
            // trailing punctuation like a period or comma (group 2).
            const listRegex = /\[(.*?)\](\s*[.,?!])?/g;

            systemText = systemText.replace(listRegex, (match, listContent, punctuation) => {
              // Clean the list content as before
              const cleanedContent = listContent.replace(/'/g, '').replace(/, /g, ' ').trim();

              // If punctuation was captured, return the content with a space before the punctuation
              if (punctuation) {
                return cleanedContent + ' ' + punctuation.trim();
              }

              // Otherwise, just return the cleaned content
              return cleanedContent;
            });

            // NOW, run the efficient replacer on the cleaned-up string.
            const finalSystemText = window.werewolfGamePlayer.playerIdReplacer(systemText);

            li.className = `moderator-announcement`;
            li.innerHTML = `
                        <cite>Moderator 
                        ${timestampHtml}</cite>
                        <div class="moderator-announcement-content ${phaseClass}">
                            <div class="msg-text">${finalSystemText.replace(/\n/g, '<br>')}</div>
                        </div>
                    `;

            const content = li.querySelector('.moderator-announcement-content');
            if (content) {
              content.style.cursor = 'pointer'; // Optional: make it look clickable
              content.onclick = (e) => {
                e.stopPropagation();
                speak(entry.allEventsIndex);
              };
            }
            break;
          case 'exile':
            const exiledPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
            li.className = `msg-entry game-event event-day`;
            let role_text = entry.role ? ` (${entry.role})` : '';
            li.innerHTML = `<cite>Exile ${timestampHtml}</cite><div class="msg-text">${exiledPlayerCap}${role_text} was exiled by vote.</div>`;
            li.style.cursor = 'pointer';
            li.onclick = (e) => {
              e.stopPropagation();
              speak(entry.allEventsIndex);
            };
            break;
          case 'elimination':
            const elimPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
            li.className = `msg-entry game-event event-night`;
            let elim_role_text = entry.role ? ` Their role was a ${entry.role}.` : '';
            li.innerHTML = `<cite>Elimination ${timestampHtml}</cite><div class="msg-text">${elimPlayerCap} was eliminated.${elim_role_text}</div>`;
            li.style.cursor = 'pointer';
            li.onclick = (e) => {
              e.stopPropagation();
              speak(entry.allEventsIndex);
            };
            break;
          case 'save':
            const savedPlayerCap = createPlayerCapsule(playerMap.get(entry.saved_player));
            li.className = `msg-entry event-night`;
            li.innerHTML = `<cite>Doctor Save ${timestampHtml}</cite><div class="msg-text">Player ${savedPlayerCap} was attacked but saved by a Doctor!</div>`;
            li.style.cursor = 'pointer';
            li.onclick = (e) => {
              e.stopPropagation();
              speak(entry.allEventsIndex);
            };
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
                            <cite>
                                <span> <span>${voter.name}</span>
                                    ${voter.display_name && voter.name !== voter.display_name ? `<span class="display-name">${voter.display_name}</span>` : ''}
                                </span> ${reasoningToggleHtml}
                                ${timestampHtml}
                            </cite>
                            <div class="balloon">
                                <div class="balloon-text">${voterCap} votes to exile ${voteTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
            const vote_balloon = li.querySelector('.balloon');
            if (vote_balloon) {
              vote_balloon.onclick = (e) => {
                e.stopPropagation();
                speak(entry.allEventsIndex);
              };
            }
            break;
          case 'timeout':
            const to_voter = playerMap.get(entry.actor_id);
            if (!to_voter) return;
            const to_voterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
            li.className = `chat-entry event-day`;
            li.innerHTML = `
                        <img src="${to_voter.thumbnail}" alt="${to_voter.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${to_voter.name} ${reasoningToggleHtml} ${timestampHtml}</cite>
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
                            <cite>Werewolf Secret Vote ${reasoningToggleHtml} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${nightVoterCap} votes to eliminate ${nightVoteTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
            const nvote_balloon = li.querySelector('.balloon');
            if (nvote_balloon) {
              nvote_balloon.onclick = (e) => {
                e.stopPropagation();
                speak(entry.allEventsIndex);
              };
            }
            break;
          case 'game_over':
            const winnersText = entry.winners.map((p) => createPlayerCapsule(playerMap.get(p))).join(' ');
            const losersText = entry.losers.map((p) => createPlayerCapsule(playerMap.get(p))).join(' ');
            li.className = `msg-entry game-win ${phaseClass}`;
            li.innerHTML = `
                        <cite>Game Over ${timestampHtml}</cite>
                        <div class="msg-text">
                            <div>The <strong>${entry.winner}</strong> team has won!</div><br>
                            <div><strong>Winning Team:</strong> ${winnersText}</div>
                            <div><strong>Losing Team:</strong> ${losersText}</div>
                        </div>
                    `;
            li.style.cursor = 'pointer';
            li.onclick = (e) => {
              e.stopPropagation();
              speak(entry.allEventsIndex);
            };
            break;
        }
        if (li.innerHTML) logUl.appendChild(li);
      });
    }

    container.appendChild(logUl);
    logUl.scrollTop = logUl.scrollHeight;

    const globalToggle = container.querySelector('#global-reasoning-toggle');
    if (globalToggle) {
      // Initialize global state if not present
      if (window.werewolfGamePlayer.isReasoningMode === undefined) {
        window.werewolfGamePlayer.isReasoningMode = false;
      }

      globalToggle.addEventListener('click', (event) => {
        event.stopPropagation();

        // --- 1. Toggle 2D Log Reasoning (Original Behavior) ---
        const reasoningTexts = logUl.querySelectorAll('.reasoning-text');
        if (reasoningTexts.length > 0) {
          // Determine if we should show or hide all. If any are visible, we hide all. Otherwise, show all.
          const shouldShow = ![...reasoningTexts].some((el) => el.classList.contains('visible'));

          reasoningTexts.forEach((el) => {
            el.classList.toggle('visible', shouldShow);
          });
        }

        // --- 2. Toggle 3D Bubble Reasoning (New Behavior) ---

        // Toggle the global state
        window.werewolfGamePlayer.isReasoningMode = !window.werewolfGamePlayer.isReasoningMode;
        const isGlobalReasoningOn = window.werewolfGamePlayer.isReasoningMode;

        // Find all active 3D UI containers in the document
        // We search the whole document because the 3D UI is rendered by Three.js
        const allPlayerUIs = document.querySelectorAll('.player-ui-container.chat-active');

        allPlayerUIs.forEach((uiElement) => {
          const reasoningEl = uiElement.querySelector('.bubble-reasoning');
          // Check if the reasoning element has content
          const hasReasoning = reasoningEl && (reasoningEl.innerHTML || reasoningEl.textContent);

          if (isGlobalReasoningOn && hasReasoning) {
            uiElement.classList.add('show-reasoning');
          } else {
            uiElement.classList.remove('show-reasoning');
          }
        });
      });
    }

    const globalAudioToggle = container.querySelector('#global-audio-toggle');
    if (globalAudioToggle) {
      globalAudioToggle.addEventListener('click', (event) => {
        event.stopPropagation();
        if (globalAudioToggle.classList.contains('disabled')) return;

        const audioState = window.kaggleWerewolf; // Get the state
        const wasEnabled = audioState.isAudioEnabled;

        if (wasEnabled) {
          // --- DISABLING ---
          audioState.isAudioEnabled = false;
          globalAudioToggle.classList.remove('enabled');
          globalAudioToggle.innerHTML = '&#x1F507;'; // Muted icon

          stopAndClearAudio(); // Stop current playback
          audioState.isPaused = true; // Ensure it stays paused

          // Update left-panel pause button
        } else {
          // --- ENABLING ---
          audioState.isAudioEnabled = true;
          globalAudioToggle.classList.add('enabled');
          globalAudioToggle.innerHTML = '&#x1F50A;'; // Speaker icon

          // Activate audio context if this is the very first time
          if (!audioState.audioContextActivated) {
            const audio = new Audio(
              'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA'
            );
            audio.play().catch((e) => console.warn('Audio context activation failed:', e));
            audioState.audioContextActivated = true; // Set new flag
          }

          // Check if there was a pending playback request (e.g., from play button)
          if (audioState.pendingPlaybackRequest) {
            const { startIndex, isContinuous } = audioState.pendingPlaybackRequest;
            audioState.pendingPlaybackRequest = null;
            // We are now enabled, so this will work.
            playAudioFrom(startIndex, isContinuous);
          }
        }
      });
    }

    const speedSlider = container.querySelector('#playback-speed');
    const speedLabel = container.querySelector('#speed-label');

    if (speedSlider) {
      speedSlider.addEventListener('input', (e) => {
        const newRate = parseFloat(e.target.value);
        setPlaybackRate(newRate);
        if (speedLabel) speedLabel.textContent = newRate.toFixed(1) + 'x';
      });
    }
  }

  export function renderPlayerList(container, gameState, actingPlayerName) {
    container.innerHTML = '<h1>Players</h1>';
    const listContainer = document.createElement('div');
    listContainer.id = 'player-list-container';
    const playerUl = document.createElement('ul');
    playerUl.id = 'player-list';

    gameState.players.forEach((player) => {
      const li = document.createElement('li');
      li.className = 'player-card';
      if (!player.is_alive) li.classList.add('dead');
      if (player.name === actingPlayerName) li.classList.add('active');

      let roleDisplay = player.role;
      if (player.role === 'Werewolf') {
        roleDisplay = `&#x1F43A; ${player.role}`;
      } else if (player.role === 'Doctor') {
        roleDisplay = `&#x1FA7A; ${player.role}`;
      } else if (player.role === 'Seer') {
        roleDisplay = `&#x1F52E; ${player.role}`;
      } else if (player.role === 'Villager') {
        roleDisplay = `&#x1F9D1; ${player.role}`;
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

    const audioControls = document.createElement('div');
    audioControls.className = 'audio-controls';
    audioControls.innerHTML = `
        <label for="playback-speed">Audio Speed: <span id="speed-label">${audioState.playbackRate.toFixed(1)}</span>x</label>
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
            <input type="range" id="playback-speed" min="0.5" max="2.5" step="0.1" value="${audioState.playbackRate}" style="flex-grow: 1;">
        </div>
    `;
    container.appendChild(audioControls);

    const speedSlider = audioControls.querySelector('#playback-speed');
    const speedLabel = audioControls.querySelector('#speed-label');

    speedSlider.addEventListener('input', (e) => {
      const newRate = parseFloat(e.target.value);
      setPlaybackRate(newRate);
      speedLabel.textContent = newRate.toFixed(1);
    });
  }