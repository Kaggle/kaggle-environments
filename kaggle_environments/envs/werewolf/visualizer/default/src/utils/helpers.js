import { applyTranscriptOverrides } from './transcriptUtils.js';

// Re-implemented locally to fix HTML artifact bugs in core
// (Use 2-pass placeholder strategy to avoid replacing text inside generated HTML attributes)
export function createPlayerCapsule(player) {
  if (!player) return '';
  const nameToShow = player.display_name || player.name;
  const thumbnailSrc = player.thumbnail || '';
  // Escape quotes in the name to prevent attribute breakage
  const safeName = (player.name || '').replace(/"/g, '&quot;');
  return `<span class="player-capsule" title="${safeName}">
    <img src="${thumbnailSrc}" class="capsule-avatar" alt="${safeName}" onerror="handleThumbnailError(this)">
    <span class="capsule-name">${nameToShow}</span>
  </span>`;
}

export function createNameReplacer(playerMap, format = 'text') {
  const textCache = new Map();
  // Unique placeholder prefix (Markdown safe - no leading underscores)
  const PLACEHOLDER_PREFIX = `KGL_CAP_${Math.floor(Math.random() * 1000000)}_`;

  const sortedPlayerReplacements = [...playerMap.keys()]
    .sort((a, b) => b.length - a.length)
    .map((characterName, index) => {
      const player = playerMap.get(characterName);
      if (!player) return null;

      const displayName = player.display_name || characterName;
      if (format === 'text' && displayName === characterName) return null;

      const replacementHtml = format === 'html'
        ? createPlayerCapsule(player)
        : displayName;

      // Wrap placeholder in unique characters to prevent partial prefix replacement during expansion
      // (e.g. preventing KGL_CAP_1 from matching inside KGL_CAP_10)
      const placeholder = `|${PLACEHOLDER_PREFIX}${index}|`;
      const escapedName = characterName.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');

      // Regex handling "Player Name" simplification and word boundaries
      // Match inside quotes (text) but try to avoid breaking attributes if simple replacer is used.
      // Stronger boundary: include dots and hyphens in the exclusion to prevent prefix matching for versions/dots.
      const regex = new RegExp(
        `(^|[^\\w.-])(?:Player\\s*)?(${escapedName})(\\.?)(?![\\w.-])`,
        'gi'
      );

      return { regex, placeholder, replacementHtml };
    })
    .filter(r => r !== null);

  const replaceToPlaceholders = (text) => {
    if (!text) return '';
    let newText = text;
    for (const { regex, placeholder } of sortedPlayerReplacements) {
      newText = newText.replace(regex, `$1${placeholder}$3`);
    }
    return newText;
  };

  const expandPlaceholders = (text) => {
    if (!text) return '';
    let newText = text;
    for (const { placeholder, replacementHtml } of sortedPlayerReplacements) {
      // Global replace of the placeholder string
      newText = newText.split(placeholder).join(replacementHtml);
    }
    return newText;
  };

  const replaceNames = function (text) {
    if (!text) return '';
    if (textCache.has(text)) return textCache.get(text);

    let newText = replaceToPlaceholders(text);
    newText = expandPlaceholders(newText);

    textCache.set(text, newText);
    return newText;
  };

  replaceNames.replaceToPlaceholders = replaceToPlaceholders;
  replaceNames.expandPlaceholders = expandPlaceholders;

  return replaceNames;
}

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

// Dark purple background (#2d1b4e) with white question mark
export const FALLBACK_THUMBNAIL_IMG = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCI+PHJlY3Qgd2lkdGg9IjEwMCIgaGVpZ2h0PSIxMDAiIGZpbGw9IiMyZDFiNGUiLz48dGV4dCB4PSI1MCIgeT0iNzAiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSI2MCIgZmlsbD0id2hpdGUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtd2VpZ2h0PSJib2xkIj4/PC90ZXh0Pjwvc3ZnPg==";

// Global handler for thumbnail errors
window.handleThumbnailError = function (img) {
  if (img && img.src !== FALLBACK_THUMBNAIL_IMG) {
    img.onerror = null; // Prevent infinite loop
    img.src = FALLBACK_THUMBNAIL_IMG;
  }
};

/**
 * Creates a memoized function to replace player IDs with HTML capsules.
 * Convenience wrapper around createNameReplacer with 'html' format.
 * @param {Map<string, object>} playerMap - A map from player ID to player object.
 * @returns {function(string): string} A function that takes text and returns it with player IDs replaced.
 */
export function createPlayerIdReplacer(playerMap) {
  return createNameReplacer(playerMap, 'html');
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
      if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
        window.werewolfThreeJs.demo.resetCameraView();
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
      if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
        // Get the current widths of the UI panels
        const leftPanel = container.closest('.left-panel'); // Assuming container is in left panel
        const eventPanel = document.querySelector('.event-panel');
        const leftPanelWidth = leftPanel ? leftPanel.offsetWidth : 0;
        const eventPanelWidth = eventPanel ? eventPanel.offsetWidth : 0;

        // Pass the panel widths to the focus function
        window.werewolfThreeJs.demo.focusOnPlayer(player.name, leftPanelWidth, eventPanelWidth);
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
    const name_to_display = player.display_name || player.name;
    const player_name_element = `<div class="player-name" title="${player.name}">${name_to_display}</div>`;

    li.innerHTML = `
            <div class="avatar-container">
                <img src="${player.thumbnail}" alt="${player.name}" class="avatar" onerror="handleThumbnailError(this)">
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

export function updateEventLog(container, gameState, playerMap, onSpeak) {
  const audioState = window.kaggleWerewolf || { hasAudioTracks: false, isAudioEnabled: false, playbackRate: 1.0 };
  const audioToggleDisabled = !audioState.hasAudioTracks;
  const audioToggleEnabled = audioState.isAudioEnabled && !audioToggleDisabled;
  const audioToggleTitle = audioToggleDisabled ? 'Audio Not Available' : 'Toggle Audio';
  const audioToggleIcon = audioToggleEnabled ? '&#x1F50A;' : '&#x1F507;'; // Speaker vs Muted
  const audioToggleClasses = `audio-toggle-btn ${audioToggleDisabled ? 'disabled' : ''} ${audioToggleEnabled ? 'enabled' : ''}`;

  if (!container.querySelector('h1')) {
    container.innerHTML = `
            <h1>
                <span>Events</span>
                <button id="reset-view-btn" class="reset-view-btn" title="Reset Camera View">
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
        const isCollapsed = container.classList.toggle('collapsed');
        // Also toggle parent class for subtitle visibility
        const grandparent = container.closest('.werewolf-parent');
        if (grandparent) {
          if (isCollapsed) {
            grandparent.classList.remove('left-panel-visible');
          } else {
            grandparent.classList.add('left-panel-visible');
          }
        }
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
    resetButton.onclick = (e) => {
      e.stopPropagation();
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
        let reasoningText = entry.reasoning;
        // Apply player ID replacement to reasoning text so that e.g. "p0" becomes "Bot (1)"
        if (window.werewolfGamePlayer && window.werewolfGamePlayer.playerIdReplacer) {
          reasoningText = window.werewolfGamePlayer.playerIdReplacer(reasoningText);
        }
        reasoningHtml = `<div class="reasoning-text" id="${reasoningId}">"${reasoningText}"</div>`;
        reasoningToggleHtml = `<span class="reasoning-toggle${window.werewolfGamePlayer.isReasoningMode ? ' enabled' : ''}" title="Show/Hide Reasoning" onclick="event.stopPropagation(); this.classList.toggle('enabled'); document.getElementById('${reasoningId}').classList.toggle('visible')">
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
                        <img src="${speaker.thumbnail}" alt="${speaker.name}" class="chat-avatar" onerror="handleThumbnailError(this)">
                        <div class="message-content">
                            <cite>
                                 <span> <span>${speaker.display_name || speaker.name}</span>
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
          if (balloon && onSpeak) {
            balloon.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
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
                        <img src="${seerInspector.thumbnail}" alt="${seerInspector.name}" class="chat-avatar" onerror="handleThumbnailError(this)">
                        <div class="message-content">
                            <cite>Seer Secret Inspect ${reasoningToggleHtml} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${seerCap} chose to inspect ${seerTargetCap}'s role.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
          const seer_balloon = li.querySelector('.balloon');
          if (seer_balloon && onSpeak) {
            seer_balloon.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
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
                        <img src="${seerResultViewer.thumbnail}" alt="${seerResultViewer.name}" class="chat-avatar" onerror="handleThumbnailError(this)">
                        <div class="message-content">
                            <cite>Seer Inspect Result ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${seerCap_} saw ${seerResultTargetCap}'s ${resultString}.</div>
                            </div>
                        </div>
                    `;
          const seer_balloon_ = li.querySelector('.balloon');
          if (seer_balloon_ && onSpeak) {
            seer_balloon_.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
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
                        <img src="${doctor.thumbnail}" alt="${doctor.name}" class="chat-avatar" onerror="handleThumbnailError(this)">
                        <div class="message-content">
                            <cite>Doctor Secret Heal ${reasoningToggleHtml} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${docCap} chose to heal ${docTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
          const dr_balloon = li.querySelector('.balloon');
          if (dr_balloon && onSpeak) {
            dr_balloon.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
            };
          }
          break;
        case 'system':
          if (entry.text && entry.text.includes('has begun')) return;

          let systemText = entry.text || '';

          const listRegex = /\\\[(.*?)\\\](\\s*[.,?!])?/g;

          systemText = systemText.replace(listRegex, (match, listContent, punctuation) => {
            const cleanedContent = listContent.replace(/'/g, '').replace(/, /g, ' ').trim();
            if (punctuation) {
              return cleanedContent + ' ' + punctuation.trim();
            }
            return cleanedContent;
          });

          systemText = applyTranscriptOverrides(systemText);

          const finalSystemText = window.werewolfGamePlayer.playerIdReplacer(systemText);

          li.className = `moderator-announcement`;
          li.innerHTML = `
                        <cite>Moderator 
                        ${timestampHtml}</cite>
                        <div class="moderator-announcement-content ${phaseClass}">
                            <div class="msg-text">${finalSystemText.replace(/\n/g, '<br>')}
</div>
                        </div>
                    `;

          const content = li.querySelector('.moderator-announcement-content');
          if (content && onSpeak) {
            content.style.cursor = 'pointer';
            content.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
            };
          }
          break;
        case 'exile':
          const exiledPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
          li.className = `msg-entry game-event event-day`;
          let role_text = entry.role ? ` (${entry.role})` : '';
          li.innerHTML = `<cite>Exile ${timestampHtml}</cite><div class="msg-text">${exiledPlayerCap}${role_text} was exiled by vote.</div>`;
          li.style.cursor = 'pointer';
          if (onSpeak) {
            li.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
            };
          }
          break;
        case 'elimination':
          const elimPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
          li.className = `msg-entry game-event event-night`;
          let elim_role_text = entry.role ? ` Their role was a ${entry.role}.` : '';
          li.innerHTML = `<cite>Elimination ${timestampHtml}</cite><div class="msg-text">${elimPlayerCap} was eliminated.${elim_role_text}</div>`;
          li.style.cursor = 'pointer';
          if (onSpeak) {
            li.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
            };
          }
          break;
        case 'save':
          const savedPlayerCap = createPlayerCapsule(playerMap.get(entry.saved_player));
          li.className = `msg-entry event-night`;
          li.innerHTML = `<cite>Doctor Save ${timestampHtml}</cite><div class="msg-text">Player ${savedPlayerCap} was attacked but saved by a Doctor!</div>`;
          li.style.cursor = 'pointer';
          if (onSpeak) {
            li.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
            };
          }
          break;
        case 'vote':
          const voter = playerMap.get(entry.actor_id);
          if (!voter) return;
          const voterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
          const voteTargetCap = createPlayerCapsule(playerMap.get(entry.target));
          li.className = `chat-entry event-day`;
          li.innerHTML = `
                        <img src="${voter.thumbnail}" alt="${voter.name}" class="chat-avatar" onerror="handleThumbnailError(this)">
                        <div class="message-content">
                            <cite>
                                 <span> <span>${voter.display_name || voter.name}</span>
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
          if (vote_balloon && onSpeak) {
            vote_balloon.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
            };
          }
          break;
        case 'timeout':
          const to_voter = playerMap.get(entry.actor_id);
          if (!to_voter) return;
          const to_voterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
          li.className = `chat-entry event-day`;
          li.innerHTML = `
                        <img src="${to_voter.thumbnail}" alt="${to_voter.name}" class="chat-avatar" onerror="handleThumbnailError(this)">
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
                        <img src="${nightVoter.thumbnail}" alt="${nightVoter.name}" class="chat-avatar" onerror="handleThumbnailError(this)">
                        <div class="message-content">
                            <cite>Werewolf Secret Vote ${reasoningToggleHtml} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${nightVoterCap} votes to eliminate ${nightVoteTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
          const nvote_balloon = li.querySelector('.balloon');
          if (nvote_balloon && onSpeak) {
            nvote_balloon.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
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
          if (onSpeak) {
            li.onclick = (e) => {
              e.stopPropagation();
              onSpeak(entry.allEventsIndex);
            };
          }
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
    // Initialize visual state
    globalToggle.classList.toggle('enabled', window.werewolfGamePlayer.isReasoningMode);

    globalToggle.onclick = (event) => {
      event.stopPropagation();

      // --- 1. Toggle 2D Log Reasoning (Original Behavior) ---
      const reasoningTexts = logUl.querySelectorAll('.reasoning-text');
      if (reasoningTexts.length > 0) {
        // Determine if we should show or hide all. If any are visible, we hide all. Otherwise, show all.
        const shouldShow = ![...reasoningTexts].some((el) => el.classList.contains('visible'));

        reasoningTexts.forEach((el) => {
          el.classList.toggle('visible', shouldShow);
        });

        // Sync individual toggle buttons
        const individualToggles = logUl.querySelectorAll('.reasoning-toggle');
        individualToggles.forEach(toggle => {
          toggle.classList.toggle('enabled', shouldShow);
        });
      }

      // --- 2. Toggle Global Reasoning State ---
      window.werewolfGamePlayer.isReasoningMode = !window.werewolfGamePlayer.isReasoningMode;
      const isGlobalReasoningOn = window.werewolfGamePlayer.isReasoningMode;

      // Toggle visual state
      globalToggle.classList.toggle('enabled', isGlobalReasoningOn);

      // --- 3. Toggle 3D Bubble Reasoning (Legacy) ---
      const allPlayerUIs = document.querySelectorAll('.player-ui-container.chat-active');
      allPlayerUIs.forEach((uiElement) => {
        const reasoningEl = uiElement.querySelector('.bubble-reasoning');
        const hasReasoning = reasoningEl && (reasoningEl.innerHTML || reasoningEl.textContent);
        if (isGlobalReasoningOn && hasReasoning) {
          uiElement.classList.add('show-reasoning');
        } else {
          uiElement.classList.remove('show-reasoning');
        }
      });

      // --- 4. Toggle Cinematic Subtitle Reasoning ---
      const subtitleContainer = document.querySelector('.cinematic-subtitle-container');
      if (subtitleContainer) {
        if (isGlobalReasoningOn) {
          subtitleContainer.classList.add('show-reasoning');
        } else {
          subtitleContainer.classList.remove('show-reasoning');
        }
      }
    };
  }

  const globalAudioToggle = container.querySelector('#global-audio-toggle');
  if (globalAudioToggle) {
    globalAudioToggle.onclick = (event) => {
      event.stopPropagation();
      if (globalAudioToggle.classList.contains('disabled')) return;

      const audioState = window.kaggleWerewolf; // Get the state
      const wasEnabled = audioState.isAudioEnabled;

      if (wasEnabled) {
        // --- DISABLING ---
        audioState.isAudioEnabled = false;
        globalAudioToggle.classList.remove('enabled');
        globalAudioToggle.innerHTML = '&#x1F507;'; // Muted

        // Use custom event
        const event = new CustomEvent('audio-toggle', { detail: { enabled: false } });
        window.dispatchEvent(event);

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
          audioState.audioContextActivated = true;
        }

        const event = new CustomEvent('audio-toggle', { detail: { enabled: true } });
        window.dispatchEvent(event);
      }
    };
  }

  const speedSlider = container.querySelector('#playback-speed');
  const speedLabel = container.querySelector('#speed-label');

  if (speedSlider) {
    speedSlider.onclick = (e) => {
      e.stopPropagation();
    };

    speedSlider.oninput = (e) => {
      const newRate = parseFloat(e.target.value);
      // Use custom event for speed change (legacy audio-speed)
      const audioEvent = new CustomEvent('audio-speed', { detail: { rate: newRate } });
      window.dispatchEvent(audioEvent);

      // Also notify the ReplayVisualizer so it can sync with parent
      const replayerEvent = new CustomEvent('replayer-speed', { detail: { rate: newRate, fromReplayer: false } });
      window.dispatchEvent(replayerEvent);

      if (speedLabel) speedLabel.textContent = newRate.toFixed(1) + 'x';
    };

    // Listen for speed changes from ReplayVisualizer and sync the slider
    window.addEventListener('replayer-speed', (e) => {
      if (e.detail.fromReplayer && speedSlider) {
        const newRate = e.detail.rate;
        // Only update if different to avoid loops
        if (parseFloat(speedSlider.value) !== newRate) {
          speedSlider.value = newRate.toString();
          if (speedLabel) speedLabel.textContent = newRate.toFixed(1) + 'x';
          // Also update the audioState
          const audioEvent = new CustomEvent('audio-speed', { detail: { rate: newRate } });
          window.dispatchEvent(audioEvent);
        }
      }
    });
  }
}

export function getPermutation(items, seed) {
  const m = 2147483648n; // 2^31
  const a = 1103515245n;
  const c = 12345n;
  let shuffledItems = [...items];
  let n = shuffledItems.length;
  let currentSeed = BigInt(seed);

  for (let i = n - 1; i > 0; i--) {
    currentSeed = (a * currentSeed + c) % m;
    // precise modulo with BigInt
    let j = Number(currentSeed % BigInt(i + 1));
    [shuffledItems[i], shuffledItems[j]] = [shuffledItems[j], shuffledItems[i]];
  }
  return shuffledItems;
}

export function shuffleIds(agentsConfig, seed) {
  if (!agentsConfig || !seed) return [];
  const ids = agentsConfig.map(a => a.id);
  // seed + 123 is used in python engine
  return getPermutation(ids, seed + 123);
}