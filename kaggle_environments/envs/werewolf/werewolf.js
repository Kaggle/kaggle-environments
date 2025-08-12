function renderer({
  environment,
  step,
  parent,
  height = 700, // Default height
  width = 1100, // Default width
}) {
  // --- THREE.js Scene Setup (Singleton Pattern) ---
  if (!window.werewolfThreeJs) {
    window.werewolfThreeJs = {
      initialized: false,
      scene: null,
      camera: null,
      renderer: null,
      model: null,
      ambientLight: null,
      directionalLight: null,
    };
  }
  const threeState = window.werewolfThreeJs;

  function initThreeJs() {
    if (threeState.initialized) {
        if (threeState.renderer && parent && !parent.contains(threeState.renderer.domElement)) {
            parent.appendChild(threeState.renderer.domElement);
        }
        return;
    }

    if (typeof THREE === 'undefined' || (typeof THREE.GLTFLoader === 'undefined')) {
      const scriptId = 'threejs-script';
      if (document.getElementById(scriptId)) return; // Script already requested

      const script = document.createElement('script');
      script.id = scriptId;
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
      script.onload = () => {
        const loaderScript = document.createElement('script');
        loaderScript.id = 'gltf-loader-script';
        loaderScript.src = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js';
        loaderScript.onload = setupScene;
        document.head.appendChild(loaderScript);
      };
      document.head.appendChild(script);
    } else {
      setupScene();
    }
  }

  function setupScene() {
    if (threeState.initialized) return;

    threeState.scene = new THREE.Scene();
    threeState.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    threeState.renderer = new THREE.WebGLRenderer({ antialias: true }); // No alpha needed
    threeState.renderer.setSize(width, height);

    const threeCanvas = threeState.renderer.domElement;
    threeCanvas.style.position = 'absolute';
    threeCanvas.style.top = '0';
    threeCanvas.style.left = '0';
    threeCanvas.style.zIndex = '0'; // Behind UI
    parent.appendChild(threeCanvas);

    threeState.ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
    threeState.scene.add(threeState.ambientLight);
    threeState.directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    threeState.directionalLight.position.set(5, 10, 7.5);
    threeState.scene.add(threeState.directionalLight);

    const loader = new THREE.GLTFLoader();
//    loader.load(
//      'assets/low_poly_medieval_windmill/scene.gltf',
//      function (gltf) {
//        threeState.model = gltf.scene;
//        threeState.model.scale.set(0.5, 0.5, 0.5); // Adjust scale for the new model
//        threeState.model.position.set(0, -2, 0); // Adjust position
//        threeState.scene.add(threeState.model);
//      },
//      undefined,
//      function(error) {
//        console.error('An error occurred loading the model:', error);
//      }
//    );
//
//    threeState.camera.position.z = 5;
//    threeState.initialized = true;
//    animate();
  }

  function animate() {
    if (!threeState.initialized) return;
    requestAnimationFrame(animate);
    if (threeState.model) {
      threeState.model.rotation.y += 0.005;
    }
    threeState.renderer.render(threeState.scene, threeState.camera);
  }

  function updateBackground(isNight) {
      if (!threeState.initialized) return;
      if (isNight) {
          threeState.scene.background = new THREE.Color(0x2c3e50); // Night color from CSS var
          threeState.ambientLight.intensity = 0.5;
          threeState.directionalLight.intensity = 0.5;
      } else {
          threeState.scene.background = new THREE.Color(0x3498db); // Day color from CSS var
          threeState.ambientLight.intensity = 1.0;
          threeState.directionalLight.intensity = 1.0;
      }
  }

  // --- CSS Loading System ---
  function loadCSS() {
    // Check if CSS is already loaded
    const existingLink = document.querySelector('link[href*="werewolf.css"]');
    if (existingLink) return;
    
    // Create and append CSS link
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = './kaggle_environments/envs/werewolf/werewolf.css';
    
    // Add error handling for CSS loading
    link.onerror = () => {
      console.warn('Could not load werewolf.css from relative path, trying absolute path');
      link.href = '/kaggle_environments/envs/werewolf/werewolf.css';
    };
    
    document.head.appendChild(link);
  }

  // --- TTS Management ---
  const audioMap = window.AUDIO_MAP || {};

  if (!window.kaggleWerewolf) {
      window.kaggleWerewolf = {
          audioQueue: [],
          isAudioPlaying: false,
          isAudioEnabled: false,
          isPaused: false,
          lastPlayedStep: -1,
          audioPlayer: new Audio(),
          playbackRate: 1.0,
      };
  }
  const audioState = window.kaggleWerewolf;

  function togglePause() {
      audioState.isPaused = !audioState.isPaused;
      const pauseButton = parent.querySelector('#pause-audio');
      if (pauseButton) {
          pauseButton.classList.toggle('paused', audioState.isPaused);
          pauseButton.classList.toggle('playing', !audioState.isPaused);
      }
      if (!audioState.isPaused && !audioState.isAudioPlaying) {
          playNextInQueue();
      } else if (audioState.isPaused && audioState.isAudioPlaying) {
          audioState.audioPlayer.pause();
      } else if (!audioState.isPaused && audioState.isAudioPlaying) {
          audioState.audioPlayer.play();
      }
  }

  function setPlaybackRate(rate) {
      audioState.playbackRate = rate;
      if (audioState.isAudioPlaying) {
          audioState.audioPlayer.playbackRate = rate;
      }
  }

  function playNextInQueue() {
      if (audioState.isPaused || audioState.isAudioPlaying || audioState.audioQueue.length === 0 || !audioState.isAudioEnabled) {
          return;
      }

      const previouslyPlaying = parent.querySelector('.playing-audio');
      if (previouslyPlaying) {
          previouslyPlaying.classList.remove('playing-audio');
      }

      audioState.isAudioPlaying = true;
      const event = audioState.audioQueue.shift();
      const audioKey = event.speaker === 'moderator' ? `moderator:${event.message}` : `${event.speaker}:${event.message}`;
      const audioPath = audioMap[audioKey];

      const elementToHighlight = parent.querySelector(`[data-audio-key="${audioKey}"]`);
      if (elementToHighlight) {
          elementToHighlight.classList.add('playing-audio');
      }

      if (audioPath) {
          audioState.audioPlayer.src = audioPath;
          audioState.audioPlayer.playbackRate = audioState.playbackRate;
          audioState.audioPlayer.onended = () => {
              if (elementToHighlight) elementToHighlight.classList.remove('playing-audio');
              audioState.isAudioPlaying = false;
              if (!audioState.isPaused) {
                playNextInQueue();
              }
          };
          audioState.audioPlayer.onerror = () => {
              console.error("Audio playback failed for key:", audioKey);
              if (elementToHighlight) elementToHighlight.classList.remove('playing-audio');
              audioState.isAudioPlaying = false;
              playNextInQueue();
          };
          audioState.audioPlayer.play().catch(e => {
              console.error("Audio playback failed:", e);
              if (elementToHighlight) elementToHighlight.classList.remove('playing-audio');
              audioState.isAudioPlaying = false;
              playNextInQueue();
          });
      } else {
          console.warn(`No audio found for key: "${audioKey}"`);
          if (elementToHighlight) elementToHighlight.classList.remove('playing-audio');
          audioState.isAudioPlaying = false;
          playNextInQueue();
      }
  }

  // --- Audio state activation ---
  if (!parent.dataset.audioListenerAttached) {
      parent.dataset.audioListenerAttached = 'true';
      parent.addEventListener('click', () => {
          if (!audioState.isAudioEnabled) {
              audioState.isAudioEnabled = true;
              const audio = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
              audio.play().catch(e => console.warn("Audio context activation failed:", e));
              playNextInQueue();
          }
      }, { once: true });
  }

  function getAudioEventForEntry(entry) {
    let audioEvent = null;
    if (entry.type === 'chat') {
        audioEvent = { message: entry.message, speaker: entry.speaker };
    } else if (entry.type === 'system') {
        const text = (entry.text || '').toLowerCase();
        let audioKey = null;
        if (text.includes('night') && text.includes('begins')) {
            audioKey = 'night_begins';
        } else if (text.includes('day') && text.includes('begins')) {
            audioKey = 'day_begins';
        } else if (text.includes('discussion')) {
            audioKey = 'discussion_begins';
        } else if (text.includes('voting phase begins')) {
            audioKey = 'voting_begins';
        }
        if (audioKey) {
            audioEvent = { message: audioKey, speaker: 'moderator' };
        }
    } else if (entry.type === 'exile') {
        const message = `Player ${entry.name} was exiled by vote. Their role was a ${entry.role}.`;
        audioEvent = { message: message, speaker: 'moderator' };
    } else if (entry.type === 'elimination') {
        const message = `Player ${entry.name} was eliminated. Their role was a ${entry.role}.`;
        audioEvent = { message: message, speaker: 'moderator' };
    } else if (entry.type === 'save') {
        const message = `Player ${entry.saved_player} was attacked but saved by a Doctor!`;
        audioEvent = { message: message, speaker: 'moderator' };
    } else if (entry.type === 'game_over') {
        const message = `The game is over. The ${entry.winner} team has won!`;
        audioEvent = { message: message, speaker: 'moderator' };
    }
    return audioEvent;
  }

  function speakFrom(startIndex, logEntries) {
      if (!audioState.isAudioEnabled) {
          audioState.isAudioEnabled = true;
          const audio = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
          audio.play().catch(e => console.warn("Audio context activation failed:", e));
      }

      // Stop current playback and clear highlight
      if (audioState.isAudioPlaying) {
          audioState.audioPlayer.pause();
          audioState.isAudioPlaying = false;
          const previouslyPlaying = parent.querySelector('.playing-audio');
          if (previouslyPlaying) {
              previouslyPlaying.classList.remove('playing-audio');
          }
      }

      audioState.audioQueue = [];
      if (audioState.isPaused) {
          audioState.isPaused = false;
          const pauseButton = parent.querySelector('#pause-audio');
          if (pauseButton) {
              pauseButton.classList.remove('paused');
              pauseButton.classList.add('playing');
          }
      }

      const eventsToQueue = [];
      for (let i = startIndex; i < logEntries.length; i++) {
          const entry = logEntries[i];
          const audioEvent = getAudioEventForEntry(entry);
          if (audioEvent) {
              eventsToQueue.push(audioEvent);
          }
      }

      audioState.audioQueue = eventsToQueue;
      playNextInQueue();
  }

  // --- Enhanced Interactive Features ---
  
  function showNotification(message, type = 'info', duration = 3000) {
      const toast = document.createElement('div');
      toast.className = `notification-toast ${type}`;
      toast.textContent = message;
      
      document.body.appendChild(toast);
      
      // Trigger animation
      setTimeout(() => toast.classList.add('show'), 100);
      
      // Remove after duration
      setTimeout(() => {
          toast.classList.remove('show');
          setTimeout(() => {
              if (document.body.contains(toast)) {
                  document.body.removeChild(toast);
              }
          }, 300);
      }, duration);
  }
  
  function createParticleEffect(element, count = 8) {
      const rect = element.getBoundingClientRect();
      const container = document.createElement('div');
      container.className = 'particle-effect';
      container.style.left = rect.left + rect.width / 2 + 'px';
      container.style.top = rect.top + rect.height / 2 + 'px';
      
      for (let i = 0; i < count; i++) {
          const particle = document.createElement('div');
          particle.className = 'particle';
          
          const angle = (360 / count) * i;
          const distance = 50 + Math.random() * 30;
          const x = Math.cos(angle * Math.PI / 180) * distance;
          const y = Math.sin(angle * Math.PI / 180) * distance;
          
          particle.style.left = '0px';
          particle.style.top = '0px';
          particle.style.transform = `translate(${x}px, ${y}px)`;
          
          container.appendChild(particle);
      }
      
      document.body.appendChild(container);
      
      // Remove after animation
      setTimeout(() => {
          if (document.body.contains(container)) {
              document.body.removeChild(container);
          }
      }, 2000);
  }
  
  
  function createBackgroundParticles() {
      if (parent.querySelector('.interactive-background')) return;
      
      const background = document.createElement('div');
      background.className = 'interactive-background';
      
      // Create floating particles
      for (let i = 0; i < 15; i++) {
          const particle = document.createElement('div');
          particle.className = 'background-particle';
          
          const size = Math.random() * 8 + 4;
          particle.style.width = size + 'px';
          particle.style.height = size + 'px';
          particle.style.left = Math.random() * 100 + '%';
          particle.style.animationDelay = Math.random() * 20 + 's';
          particle.style.animationDuration = (15 + Math.random() * 10) + 's';
          
          background.appendChild(particle);
      }
      
      parent.appendChild(background);
  }
  
  function addSoundWaveIndicator(element) {
      const soundWave = document.createElement('div');
      soundWave.className = 'sound-wave';
      
      for (let i = 0; i < 4; i++) {
          const bar = document.createElement('div');
          bar.className = 'sound-bar';
          soundWave.appendChild(bar);
      }
      
      element.style.position = 'relative';
      element.appendChild(soundWave);
      
      return soundWave;
  }
  
  function animatePlayerCard(card, animationType = 'bounce-in') {
      card.classList.add(`animate-${animationType}`);
      card.addEventListener('animationend', () => {
          card.classList.remove(`animate-${animationType}`);
      }, { once: true });
  }
  
  function addHoverEffects(element) {
      element.addEventListener('mouseenter', () => {
          element.style.transform = 'translateY(-2px) scale(1.02)';
          element.style.boxShadow = 'var(--shadow-lg)';
      });
      
      element.addEventListener('mouseleave', () => {
          element.style.transform = '';
          element.style.boxShadow = '';
      });
  }
  
  function createLoadingSpinner() {
      const spinner = document.createElement('div');
      spinner.className = 'loading-spinner';
      return spinner;
  }
  
  function addRoleBadge(avatarContainer, role) {
      if (role === 'Unknown') return;
      
      const badge = document.createElement('div');
      badge.className = `role-badge ${role.toLowerCase()}`;
      
      let badgeContent = '';
      switch(role) {
          case 'Werewolf': badgeContent = '🐺'; break;
          case 'Villager': badgeContent = '🧑'; break;
          case 'Seer': badgeContent = '🔮'; break;
          case 'Doctor': badgeContent = '🩺'; break;
      }
      
      badge.textContent = badgeContent;
      avatarContainer.appendChild(badge);
  }
  
  function addStatusIndicator(avatarContainer, isAlive) {
      const indicator = document.createElement('div');
      indicator.className = `status-indicator ${isAlive ? 'alive' : 'dead'}`;
      avatarContainer.appendChild(indicator);
  }

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
              const regex = new RegExp(`\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\b`, 'g');
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
        const regex = new RegExp(`\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\b`, 'g');
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

    // Initialize animation flag if it doesn't exist
    if (!window.werewolfAnimationsShown) {
        window.werewolfAnimationsShown = {
            playerListEntrance: false,
            panelEntrance: false
        };
    }

    gameState.players.forEach((player, index) => {
        const li = document.createElement('li');
        li.className = 'player-card';
        if (!player.is_alive) li.classList.add('dead');
        if (player.name === actingPlayerName) li.classList.add('active');

        let roleDisplay = player.role;
        if (player.role === 'Werewolf') {
            roleDisplay = `\uD83D\uDC3A ${player.role}`;
        } else if (player.role === 'Doctor') {
            roleDisplay = `\uD83E\uDE7A ${player.role}`;
        } else if (player.role === 'Seer') {
            roleDisplay = `\uD83D\uDD2E ${player.role}`;
        } else if (player.role === 'Villager') {
            roleDisplay = `\uD83E\uDDD1 ${player.role}`;
        }

        const roleText = player.role !== 'Unknown' ? `Role: ${roleDisplay}` : 'Role: Unknown';

        // Set accessibility attributes
        li.setAttribute('data-role', player.role);
        li.setAttribute('role', 'listitem');
        li.setAttribute('tabindex', '0');
        
        const statusText = player.is_alive ? 'Alive' : (player.status || 'Dead');
        const threatLevel = gameState.playerThreatLevels.get(player.name) || 0;
        const threatLabels = ['Safe', 'Uneasy', 'In Danger'];
        const threatLabel = threatLabels[Math.floor(threatLevel * 2)] || 'Safe';
        
        // Set ARIA label for the entire card
        const ariaLabel = `${player.name}, ${player.role}, ${statusText}${threatLevel > 0 ? `, threat level: ${threatLabel}` : ''}`;
        li.setAttribute('aria-label', ariaLabel);

        li.innerHTML = `
            <div class="avatar-container">
                <img src="${player.thumbnail}" alt="${player.name} avatar" class="avatar">
            </div>
            <div class="player-info">
                <div class="player-name" title="${player.name}">${player.name}</div>
                <div class="player-role">${roleText}</div>
                <div class="player-status">${statusText}</div>
            </div>
            <div class="threat-indicator"
                 style="--threat-level: ${threatLevel}"
                 data-threat-label="${threatLabel}"
                 aria-label="Threat level: ${threatLabel}"
                 role="img"></div>
        `;
        
        // Add enhanced interactive features
        const avatarContainer = li.querySelector('.avatar-container');
        addRoleBadge(avatarContainer, player.role);
        addStatusIndicator(avatarContainer, player.is_alive);
        
        // Add hover effects
        addHoverEffects(li);
        
        // Add entrance animation only once at the beginning
        if (!window.werewolfAnimationsShown.playerListEntrance) {
            setTimeout(() => animatePlayerCard(li, 'slide-in-left'), index * 100);
        }
        
        
        // Add click effects
        li.addEventListener('click', () => {
            createParticleEffect(li);
            if (player.is_alive) {
                showNotification(`Selected ${player.name}`, 'info', 2000);
            }
        });
        
        playerUl.appendChild(li);
    });

    // Mark player list entrance animation as shown
    if (!window.werewolfAnimationsShown.playerListEntrance) {
        window.werewolfAnimationsShown.playerListEntrance = true;
    }

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

    // --- Audio Controls ---
    const audioControls = document.createElement('div');
    audioControls.className = 'audio-controls';
    const pauseButtonClass = audioState.isPaused ? 'paused' : 'playing';
    audioControls.innerHTML = `
        <label for="playback-speed">Audio Speed: <span id="speed-label">${audioState.playbackRate.toFixed(1)}</span>x</label>
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
            <input type="range" 
                   id="playback-speed" 
                   min="0.5" 
                   max="2.5" 
                   step="0.1" 
                   value="${audioState.playbackRate}" 
                   style="flex-grow: 1;"
                   aria-label="Audio playback speed"
                   aria-describedby="speed-label">
            <button id="pause-audio" 
                    class="${pauseButtonClass}"
                    aria-label="${audioState.isPaused ? 'Play audio' : 'Pause audio'}"
                    aria-pressed="${!audioState.isPaused}"
                    type="button"></button>
        </div>
    `;
    container.appendChild(audioControls);

    const speedSlider = audioControls.querySelector('#playback-speed');
    const speedLabel = audioControls.querySelector('#speed-label');
    const pauseButton = audioControls.querySelector('#pause-audio');

    speedSlider.addEventListener('input', (e) => {
        const newRate = parseFloat(e.target.value);
        setPlaybackRate(newRate);
        speedLabel.textContent = newRate.toFixed(1);
    });

    pauseButton.addEventListener('click', () => {
        togglePause();
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
        logEntries.forEach((entry, index) => {
            const li = document.createElement('li');
            const audioEvent = getAudioEventForEntry(entry);
            if (audioEvent) {
                const audioKey = audioEvent.speaker === 'moderator' ? `moderator:${audioEvent.message}` : `${audioEvent.speaker}:${audioEvent.message}`;
                li.setAttribute('data-audio-key', audioKey);
            }
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
                                <div class="balloon-text">
                                    <quote>${messageText}</quote>
                                </div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    const balloon = li.querySelector('.balloon');
                    if (balloon) {
                        balloon.onclick = () => speakFrom(index, logEntries);
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
                        <cite>Moderator \u{1F4E2} ${timestampHtml}</cite>
                        <div class="moderator-announcement-content ${phaseClass}">
                            <div class="msg-text">${finalSystemText.replace(/\n/g, '<br>')}</div>
                        </div>
                    `;
                    const announcementContent = li.querySelector('.moderator-announcement-content');
                    if (announcementContent) {
                        const text = (entry.text || '').toLowerCase();
                        let audioKey = null;
                        if (text.includes('night') && text.includes('begins')) {
                            audioKey = 'night_begins';
                        } else if (text.includes('day') && text.includes('begins')) {
                            audioKey = 'day_begins';
                        } else if (text.includes('discussion')) {
                            audioKey = 'discussion_begins';
                        } else if (text.includes('voting phase begins')) {
                            audioKey = 'voting_begins';
                        }

                        if (audioKey) {
                            announcementContent.classList.add('clickable');
                            announcementContent.onclick = () => speakFrom(index, logEntries);
                        }
                    }
                    break;
                case 'exile':
                    const exiledPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-day clickable`;
                    li.innerHTML = `<cite>Exile ${timestampHtml}</cite><div class="msg-text">${exiledPlayerCap} (${entry.role}) was exiled by vote.</div>`;
                    li.onclick = () => speakFrom(index, logEntries);
                    break;
                case 'elimination':
                    const elimPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-night clickable`;
                    li.innerHTML = `<cite>Elimination ${timestampHtml}</cite><div class="msg-text">${elimPlayerCap} was eliminated. Their role was a ${entry.role}.</div>`;
                    li.onclick = () => speakFrom(index, logEntries);
                    break;
                case 'save':
                     const savedPlayerCap = createPlayerCapsule(playerMap.get(entry.saved_player));
                     li.className = `msg-entry event-night clickable`;
                     li.innerHTML = `<cite>Doctor Save ${timestampHtml}</cite><div class="msg-text">Player ${savedPlayerCap} was attacked but saved by a Doctor!</div>`;
                     li.onclick = () => speakFrom(index, logEntries);
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
                    li.className = `msg-entry game-win ${phaseClass} clickable`;
                    li.innerHTML = `
                        <cite>Game Over ${timestampHtml}</cite>
                        <div class="msg-text">
                            <div>The <strong>${entry.winner}</strong> team has won!</div><br>
                            <div><strong>Winning Team:</strong> ${winnersText}</div>
                            <div><strong>Losing Team:</strong> ${losersText}</div>
                        </div>
                    `;
                    li.onclick = () => speakFrom(index, logEntries);
                    break;
            }
            if (li.innerHTML) logUl.appendChild(li);
        });
    }

    container.appendChild(logUl);
    logUl.scrollTop = logUl.scrollHeight;
  }

    // --- Main Rendering Logic ---

    // Clean up previous UI, but not the canvas
    const oldUI = parent.querySelector('.main-container');
    if (oldUI) {
        parent.removeChild(oldUI);
    }
    const oldStyle = parent.querySelector('style');
    if (oldStyle) {
        parent.removeChild(oldStyle);
    }

    initThreeJs(); // Initialize three.js if not already done

    if (!environment || !environment.steps || environment.steps.length === 0 || step >= environment.steps.length) {
        const tempContainer = document.createElement("div");
        tempContainer.textContent = "Waiting for game data or invalid step...";
        parent.appendChild(tempContainer);
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
    let allPlayerNamesList;
    let playerThumbnails = {};

    if (firstObs && firstObs.all_player_ids) {
        allPlayerNamesList = firstObs.all_player_ids;
        playerThumbnails = firstObs.player_thumbnails || {};
    } else if (environment.configuration && environment.configuration.agents) {
        console.warn("Renderer: Initial observation missing or incomplete. Reconstructing players from configuration.");
        allPlayerNamesList = environment.configuration.agents.map(agent => agent.id);
        environment.configuration.agents.forEach(agent => {
            if (agent.id && agent.thumbnail) {
                playerThumbnails[agent.id] = agent.thumbnail;
            }
        });
    }

    if (!allPlayerNamesList || allPlayerNamesList.length === 0) {
        const tempContainer = document.createElement("div");
        tempContainer.textContent = "Waiting for game data: No players found in observation or configuration.";
        parent.appendChild(tempContainer);
        return;
    }

    gameState.players = allPlayerNamesList.map(name => ({
        name: name, is_alive: true, role: 'Unknown', team: 'Unknown', status: 'Alive',
        thumbnail: playerThumbnails[name] || `https://via.placeholder.com/40/2c3e50/ecf0f1?text=${name.charAt(0)}`
    }));
    const playerMap = new Map(gameState.players.map(p => [p.name, p]));

    // Initialize all threat levels to 0 (SAFE)
    gameState.players.forEach(p => gameState.playerThreatLevels.set(p.name, 0));

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
                if (currentObsForStep.day > 0) gameState.eventLog.push({ type: 'system', step: s, day: currentObsForStep.day, phase: 'DAY', text: `Day ${currentObsForStep.day} has begun.` });
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
                        gameState.eventLog.push({ type: 'timeout', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: actor_id, reasoning: "Timed out", timestamp: historyEvent.created_at });
                    }
                }
                return;
            }

             switch (dataEntry.data_type) {
                case 'ChatDataEntry':
                    gameState.eventLog.push({ type: 'chat', step: s, day: historyEvent.day, phase: historyEvent.phase, speaker: data.actor_id, message: data.message, reasoning: data.reasoning, timestamp, mentioned_player_ids: data.mentioned_player_ids || [] });
                    break;
                case 'DayExileVoteDataEntry':
                    gameState.eventLog.push({ type: 'vote', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'WerewolfNightVoteDataEntry':
                    gameState.eventLog.push({ type: 'night_vote', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'DoctorHealActionDataEntry':
                    gameState.eventLog.push({ type: 'doctor_heal_action', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'SeerInspectActionDataEntry':
                    gameState.eventLog.push({ type: 'seer_inspection', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'DayExileElectedDataEntry':
                    gameState.eventLog.push({ type: 'exile', step: s, day: historyEvent.day, phase: 'DAY', name: data.elected_player_id, role: data.elected_player_role_name, timestamp });
                    break;
                case 'WerewolfNightEliminationDataEntry':
                    gameState.eventLog.push({ type: 'elimination', step: s, day: historyEvent.day, phase: 'NIGHT', name: data.eliminated_player_id, role: data.eliminated_player_role_name, timestamp });
                    break;
                case 'SeerInspectResultDataEntry':
                    gameState.eventLog.push({ type: 'seer_inspection_result', step: s, day: historyEvent.day, phase: 'NIGHT', seer: data.actor_id, target: data.target_id, role: data.role, timestamp });
                    break;
                case 'DoctorSaveDataEntry':
                    gameState.eventLog.push({ type: 'save', step: s, day: historyEvent.day, phase: 'NIGHT', saved_player: data.saved_player_id, timestamp });
                    break;
                case 'GameEndResultsDataEntry':
                    gameState.gameWinner = data.winner_team;
                    const winners = gameState.players.filter(p => p.team === data.winner_team).map(p => p.name);
                    const losers = gameState.players.filter(p => p.team !== data.winner_team).map(p => p.name);
                    gameState.eventLog.push({ type: 'game_over', step: s, day: Infinity, phase: 'GAME_OVER', winner: data.winner_team, winners, losers, timestamp });
                    break;
                default:
                    if (historyEvent.entry_type === "moderator_announcement") {
                        gameState.eventLog.push({ type: 'system', step: s, day: historyEvent.day, phase: historyEvent.phase, text: historyEvent.description, timestamp, data: data});
                    }
                    break;
             }
        });
    }

    // --- Audio Playback Management ---
    if (step < audioState.lastPlayedStep) {
        audioState.audioQueue = [];
        audioState.isAudioPlaying = false;
        if (audioState.audioPlayer) {
            audioState.audioPlayer.pause();
        }
    }

    const eventsToPlay = gameState.eventLog.filter(entry =>
        entry.step > audioState.lastPlayedStep && entry.step <= step
    );

    if (eventsToPlay.length > 0) {
        eventsToPlay.forEach(entry => {
            let audioEvent = null;
            if (entry.type === 'chat') {
                audioEvent = { message: entry.message, speaker: entry.speaker };
            } else if (entry.type === 'system') {
                const text = entry.text.toLowerCase();
                if (text.includes('night') && text.includes('begins')) {
                    audioEvent = { message: 'night_begins', speaker: 'moderator' };
                } else if (text.includes('day') && text.includes('begins')) {
                    audioEvent = { message: 'day_begins', speaker: 'moderator' };
                } else if (text.includes('discussion')) {
                    audioEvent = { message: 'discussion_begins', speaker: 'moderator' };
                } else if (text.includes('voting phase begins')) {
                    audioEvent = { message: 'voting_begins', speaker: 'moderator' };
                }
            } else if (entry.type === 'exile') {
                const message = `Player ${entry.name} was exiled by vote. Their role was a ${entry.role}.`;
                audioEvent = { message: message, speaker: 'moderator' };
            } else if (entry.type === 'elimination') {
                const message = `Player ${entry.name} was eliminated. Their role was a ${entry.role}.`;
                audioEvent = { message: message, speaker: 'moderator' };
            } else if (entry.type === 'save') {
                const message = `Player ${entry.saved_player} was attacked but saved by a Doctor!`;
                audioEvent = { message: message, speaker: 'moderator' };
            } else if (entry.type === 'game_over') {
                const message = `The game is over. The ${entry.winner} team has won!`;
                audioEvent = { message: message, speaker: 'moderator' };
            }

            if (audioEvent) {
                audioState.audioQueue.push(audioEvent);
            }
        });

        if (audioState.isAudioEnabled && !audioState.isAudioPlaying) {
            playNextInQueue();
        }
    }
    audioState.lastPlayedStep = step;


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

    Object.assign(parent.style, { width: `${width}px`, height: `${height}px` });
    parent.className = 'werewolf-parent';

    // Load external CSS instead of inline styles
    loadCSS();

    const isNight = (gameState.game_state_phase || '').toLowerCase() === 'night';
    updateBackground(isNight);

    const mainContainer = document.createElement('div');
    mainContainer.className = 'main-container';
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
    
    // Add enhanced interactive features
    createBackgroundParticles();
    
    // Add sound wave indicators to playing audio elements
    const playingElements = parent.querySelectorAll('.playing-audio');
    playingElements.forEach(element => {
        if (!element.querySelector('.sound-wave')) {
            const soundWave = addSoundWaveIndicator(element);
            soundWave.classList.add('active');
        }
    });
    
    // Add entrance animations to panels only once
    if (!window.werewolfAnimationsShown.panelEntrance) {
        setTimeout(() => {
            leftPanel.classList.add('animate-slide-in-left');
            rightPanel.classList.add('animate-slide-in-right');
            window.werewolfAnimationsShown.panelEntrance = true;
        }, 100);
    }
    
    // Show game status notifications
    if (gameState.gameWinner) {
        showNotification(`🎉 ${gameState.gameWinner} team wins!`, 'success', 5000);
    } else if (gameState.phase === 'NIGHT') {
        showNotification('🌙 Night phase - Roles take action', 'info', 3000);
    } else if (gameState.phase === 'DAY') {
        showNotification('☀️ Day phase - Discussion and voting', 'info', 3000);
    }
}