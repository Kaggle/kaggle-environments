function renderer(context) {
  const {
    environment,
    step,
    parent,
    height = 1000,
    width = 1500
  } = context;

  if (!parent.id) {
      parent.id = 'werewolf-renderer-parent-' + Math.random().toString(36).substring(2, 9);
  }
  const parentId = parent.id;

  const systemEntryTypeSet = new Set([
        'moderator_announcement',
        'elimination',
        'vote_request',
        'heal_request',
        'heal_result',
        'inspect_request',
        'inspect_result',
        'bidding_info',
        'bid_result',
        'day_start',
        'night_start'
  ]);

  if (!window.werewolfGamePlayer) {
    window.werewolfGamePlayer = {
        initialized: false,
        allEvents: [],
        displayEvents: [],
        eventToKaggleStep: [],
        displayStepToAllEventsIndex: [],
        allEventsIndexToDisplayStep: [],
        originalSteps: environment.steps,
        reasoningCounter: 0,
    };
    const player = window.werewolfGamePlayer;

    const visibleEventDataTypes = new Set([
        'ChatDataEntry',
        'DayExileVoteDataEntry',
        'WerewolfNightVoteDataEntry',
        'DoctorHealActionDataEntry',
        'SeerInspectActionDataEntry',
        'DayExileElectedDataEntry',
        'WerewolfNightEliminationDataEntry',
        'SeerInspectResultDataEntry',
        'DoctorSaveDataEntry',
        'GameEndResultsDataEntry',
        'PhaseDividerDataEntry',
        'DiscussionOrderDataEntry'
    ]);

    let allEventsIndex = 0;
    let currentDisplayStep = 0;
    const processedPhaseEvents = new Set(); // This will track events within a single phase.
    (environment.info?.MODERATOR_OBSERVATION || []).forEach((stepEvents, kaggleStep) => {
        (stepEvents || []).flat().forEach(dataEntry => {
            const event = JSON.parse(dataEntry.json_str);
            const dataType = dataEntry.data_type;
            const visibleInUI = event.visible_in_ui ?? true;

            console.debug(`[RAW SEEN] Kaggle Step: ${kaggleStep}`, { dataType: dataType, event: event });

            if (!visibleInUI) {
              return;
            }

            // 1. Reset our memory on key phase-changing events.
            if (event.event_name === 'day_start' || event.event_name === 'night_start' || event.description?.includes('Voting phase begins')) {
                processedPhaseEvents.clear();
            }

            // 2. Generate a unique "fingerprint" based on essential event content.
            let eventFingerprint = event.description;
            const eventData = event.data;

            // 3. Check against the new fingerprint.
            if (processedPhaseEvents.has(eventFingerprint)) {
                return;
            }
            processedPhaseEvents.add(eventFingerprint);

            const isVisibleDataType = visibleEventDataTypes.has(dataType);
            const isVisibleEntryType = systemEntryTypeSet.has(event.event_name);

            if (!isVisibleDataType && !isVisibleEntryType) {
                return;
            }

            event.kaggleStep = kaggleStep;
            event.dataType = dataType;
            player.allEvents.push(event);
            player.eventToKaggleStep.push(kaggleStep);

            if (dataType !== 'PhaseDividerDataEntry') {
                player.displayEvents.push(event);
                player.displayStepToAllEventsIndex.push(allEventsIndex);
                player.allEventsIndexToDisplayStep[allEventsIndex] = currentDisplayStep;
                currentDisplayStep++;
            }
            allEventsIndex++;
        });
    });

    console.debug(`[FINAL STEP LIST]`, player.displayEvents);

    const newSteps = player.displayEvents.map((event) => {
        return player.originalSteps[event.kaggleStep];
    });

    setTimeout(() => {
        if (window.kaggle) {
            window.kaggle.environment.steps = newSteps;
        }
        window.postMessage({ setSteps: newSteps }, "*");
    }, 100); // A small delay to ensure player is ready
    player.initialized = true;
  }

  // We patch the functions on the 'context' object directly.
  if (context.__mainContext && !window.customPlayerControlsInjected) {
      const mainContext = context.__mainContext;

      if (!window.wwOriginals) {
          console.debug("DEBUG: Storing original controls for the first time.");
          window.wwOriginals = {
              setStep: mainContext.setStep,
              play: mainContext.play,
              pause: mainContext.pause,
              setPlaying: mainContext.setPlaying
          };
      }
          
      // --- Patch setStep ---
      if (mainContext.setSetStep) {
          mainContext.setSetStep(() => (newStep) => {
              console.debug(`DEBUG: [setStep] User manually set step to ${newStep}. Stopping audio.`);
              stopAndClearAudio();
              audioState.isPaused = true;
              window.wwOriginals.setStep(newStep); 
          });
      }

      // --- Patch Play ---
      mainContext.setPlay(() => (continuing) => {
        console.debug(`DEBUG: [setPlay] Play button clicked. Continuing: ${continuing}`);

        if (audioState.isAudioEnabled) {
            // --- AUDIO-DRIVEN PLAYBACK ---
            console.debug("DEBUG: [setPlay] Audio is ON. Using audio-driven playback.");
            window.wwOriginals.setPlaying(true); 
            let currentDisplayStep = context.step; 
            
            if (!continuing && !audioState.isPaused && currentDisplayStep === newSteps.length - 1) {
                currentDisplayStep = 0;
                window.wwOriginals.setStep(0); 
            }

            const allEventsIndex = window.werewolfGamePlayer.displayStepToAllEventsIndex[currentDisplayStep];
            if (allEventsIndex === undefined) {
                window.wwOriginals.setPlaying(false);
                return;
            }
            
            playAudioFrom(allEventsIndex, true);

        } else {
            // --- TIMER-DRIVEN PLAYBACK (When audio is off) ---
            console.debug("DEBUG: [setPlay] Audio is OFF. Using original Kaggle timer-based playback.");
            // This call uses the original player's setTimeout logic.
            window.wwOriginals.play(continuing);
        }
      });

      // --- Patch Pause ---
      mainContext.setPause(() => () => {
          console.debug("DEBUG: [setPause] Pause button clicked. Stopping audio.");

          // Stop the timer-based playback if it's running.
          window.wwOriginals.pause();

          window.wwOriginals.setPlaying(false); 
          audioState.isPaused = true;
          if (audioState.isAudioPlaying) {
              audioState.audioPlayer.pause();
          }
      });

      mainContext.patchesApplied = true;
  }

  // --- THREE.js Scene Setup (Singleton Pattern) ---
  if (!window.werewolfThreeJs) {
    window.werewolfThreeJs = {
      initialized: false,
      demo: null,
    };
  }
  const threeState = window.werewolfThreeJs;

  function playAudioFrom(startIndex, isContinuous = true) {
      console.debug(`DEBUG: [playAudioFrom] Called with startIndex: ${startIndex}, isContinuous: ${isContinuous}`);
      if (!audioState.isAudioEnabled) {
          console.error("DEBUG: [playAudioFrom] FAILED: Audio is not enabled.");
          return;
      }

      // This tells the main Kaggle player that playback is active.
      if (window.wwOriginals && window.wwOriginals.setPlaying) {
          window.wwOriginals.setPlaying(true);
      }
      // This updates your custom audio panel's button.

      stopAndClearAudio();
      console.debug("DEBUG: [playAudioFrom] Audio stopped and cleared.");

      if (audioState.isPaused) {
          console.debug("DEBUG: [playAudioFrom] Audio state was paused.");
          audioState.isPaused = false; // Un-pause regardless.

          // If we're at a *new* index (e.g., user clicked slider),
          // we must NOT resume. We must reload the queue.
          if (startIndex !== audioState.lastStartedIndex) {
              console.debug(`DEBUG: [playAudioFrom] New start index. Loading queue from: ${startIndex}`);
              audioState.lastStartedIndex = startIndex;
              loadQueueFrom(startIndex);
              playNextInQueue(isContinuous);
              return; // We are done.
          }

          // If we are at the *same* index, just resume from the (now empty) queue.
          // The queue will be re-filled by loadQueueFrom.
          console.debug("DEBUG: [playAudioFrom] Paused, resuming from same start index (or undefined).");
          // Fall through to load and play.
      }

      audioState.isPaused = false;
      audioState.lastStartedIndex = startIndex;
      loadQueueFrom(startIndex);
      playNextInQueue(isContinuous);
  }

  function loadQueueFrom(startIndex) {
      console.debug(`DEBUG: [loadQueueFrom] Loading queue from index: ${startIndex}`);
      if (!window.werewolfGamePlayer || !window.werewolfGamePlayer.allEvents) {
          console.error("DEBUG: [loadQueueFrom] CRITICAL: allEvents not found.");
          return;
      }
      const allEvents = window.werewolfGamePlayer.allEvents;
      const eventsToPlay = allEvents.slice(startIndex);
      console.debug(`DEBUG: [loadQueueFrom] Found ${eventsToPlay.length} potential events.`);

      audioState.audioQueue = []; // Clear previous queue

      if (eventsToPlay.length > 0) {
          eventsToPlay.forEach((entry, i) => {
              const allEventsIndex = startIndex + i;
              
              let audioEventDetails = null;
              const data = entry.data || {};
              const event_name = entry.event_name;
              const description = entry.description || '';
              const day_count = entry.day;

              // This logic is to identify if an event should have audio
              // and what the audio content is.
              switch (entry.dataType) {
                  case 'ChatDataEntry':
                      if (data.actor_id && data.actor_id !== 'moderator' && data.message) {
                          audioEventDetails = { message: data.message, speaker: data.actor_id };
                      }
                      break;
                  case 'DayExileVoteDataEntry':
                      if (data.actor_id && data.target_id) {
                          audioEventDetails = { message: `${data.actor_id} votes to exile ${data.target_id}.`, speaker: 'moderator' };
                      }
                      break;
                  case 'WerewolfNightVoteDataEntry':
                      if (data.actor_id && data.target_id) {
                          audioEventDetails = { message: `${data.actor_id} votes to eliminate ${data.target_id}.`, speaker: 'moderator' };
                      }
                      break;
                  case 'SeerInspectActionDataEntry':
                      if (data.actor_id && data.target_id) {
                          audioEventDetails = { message: `${data.actor_id} inspects ${data.target_id}.`, speaker: 'moderator' };
                      }
                      break;
                  case 'DoctorHealActionDataEntry':
                      if (data.actor_id && data.target_id) {
                          audioEventDetails = { message: `${data.actor_id} heals ${data.target_id}.`, speaker: 'moderator' };
                      }
                      break;
                  case 'DayExileElectedDataEntry':
                      if (data.elected_player_id && data.elected_player_role_name) {
                          audioEventDetails = { message: `${data.elected_player_id} was exiled by vote. Their role was a ${data.elected_player_role_name}.`, speaker: 'moderator' };
                      }
                      break;
                  case 'WerewolfNightEliminationDataEntry':
                      if (data.eliminated_player_id && data.eliminated_player_role_name) {
                          audioEventDetails = { message: `${data.eliminated_player_id} was eliminated. Their role was a ${data.eliminated_player_role_name}.`, speaker: 'moderator' };
                      }
                      break;
                  case 'DoctorSaveDataEntry':
                      if (data.saved_player_id) {
                          audioEventDetails = { message: `${data.saved_player_id} was attacked but saved by a Doctor!`, speaker: 'moderator' };
                      }
                      break;
                  case 'GameEndResultsDataEntry':
                      if (data.winner_team) {
                          audioEventDetails = { message: `The game is over. The ${data.winner_team} team has won!`, speaker: 'moderator' };
                      }
                      break;
                  case 'WerewolfNightEliminationElectedDataEntry':
                      if (data.elected_target_player_id) {
                          audioEventDetails = { message: `The werewolves have chosen to eliminate ${data.elected_target_player_id}.`, speaker: 'moderator' };
                      }
                      break;
                  case 'SeerInspectResultDataEntry':
                      if (data.role) {
                          audioEventDetails = { message: `${data.actor_id} saw ${data.target_id}'s role is ${data.role}.`, speaker: 'moderator'};
                      } else if (data.team) {
                          audioEventDetails = { message: `${data.actor_id} saw ${data.target_id}'s team is ${data.team}.`, speaker: 'moderator'};
                      }
                      break;
                  case 'DiscussionOrderDataEntry':
                      audioEventDetails = { message: description, speaker: 'moderator' };
              }

              if (!audioEventDetails && event_name === 'moderator_announcement') {
                  if (description.includes('discussion rule is')) {
                      audioEventDetails = { message: 'Discussion begins!', speaker: 'moderator' };
                  } else if (description.includes('Voting phase begins')) {
                      audioEventDetails = { message: 'Exile voting begins!', speaker: 'moderator' };
                  } else {
                    audioEventDetails = { message: entry.description, speaker: 'moderator' };
                  }
              } else if (!audioEventDetails && event_name === 'day_start') {
                  audioEventDetails = { message: `Day ${day_count} begins!`, speaker: 'moderator' };
              } else if (!audioEventDetails && event_name === 'night_start') {
                  audioEventDetails = { message: `Night ${day_count} begins!`, speaker: 'moderator' };
              }

              // Every event goes into the queue.
              audioState.audioQueue.push({
                  allEventsIndex: allEventsIndex,
                  audioEvent: audioEventDetails, // This will be null for events without audio
              });
          });
      }
      console.debug(`DEBUG: [loadQueueFrom] Loaded ${audioState.audioQueue.length} events into queue.`);
  }

  function playNextInQueue(isContinuous = true) {
      const currentParent = document.getElementById(parentId);
      if (!currentParent) {
          console.error("Werewolf renderer parent container not found in DOM, stopping playback.");
          stopAndClearAudio();
          return;
      }

      console.debug(`DEBUG: [playNextInQueue] Called. Queue length: ${audioState.audioQueue.length}. isPaused: ${audioState.isPaused}. isAudioPlaying: ${audioState.isAudioPlaying}.`);
      
      // 1. Clear any previously highlighted element
      const currentlyPlaying = currentParent.querySelector('#chat-log .now-playing');
      if (currentlyPlaying) {
          currentlyPlaying.classList.remove('now-playing');
      }

      if (audioState.isPaused || audioState.isAudioPlaying || audioState.audioQueue.length === 0 || !audioState.isAudioEnabled) {
          console.warn(`DEBUG: [playNextInQueue] Exiting early. Paused: ${audioState.isPaused}, Playing: ${audioState.isAudioPlaying}, Queue: ${audioState.audioQueue.length}, Enabled: ${audioState.isAudioEnabled}`);
          if (audioState.audioQueue.length === 0 && !audioState.isAudioPlaying) {
              console.debug("DEBUG: [playNextInQueue] Playback finished. Setting player to 'paused' state.");
              window.wwOriginals.setPlaying(false); // Stop the parent player
          }
          return;
      }
      
      audioState.isAudioPlaying = true;
      const event = audioState.audioQueue.shift();

      // This is the slider logic, it should always run
      if (event.allEventsIndex !== undefined) {
          const displayStep = window.werewolfGamePlayer.allEventsIndexToDisplayStep[event.allEventsIndex];
          console.debug(`DEBUG: [playNextInQueue] Found displayStep: ${displayStep} for event index ${event.allEventsIndex}`);
          
          if (displayStep !== undefined && window.wwOriginals && window.wwOriginals.setStep) {
              console.debug(`DEBUG: [playNextInQueue] ### ADVANCING SLIDER TO ${displayStep} ###`);
              window.wwOriginals.setStep(displayStep);

              // Use a short timeout to allow the DOM to update after the step change
              setTimeout(() => {
                  const freshParent = document.getElementById(parentId);
                  if (!freshParent) {
                      console.error(`DEBUG: Parent element not found after timeout for event index ${event.allEventsIndex}.`);
                      return;
                  }
                  const liToHighlight = freshParent.querySelector(`#chat-log li[data-all-events-index="${event.allEventsIndex}"]`);
                  console.debug(`DEBUG: [Timeout] Attempting to highlight element for index ${event.allEventsIndex}`, liToHighlight);
                  if (liToHighlight) {
                      liToHighlight.classList.add('now-playing');
                      console.debug(`DEBUG: [Timeout] Successfully added .now-playing to element for index ${event.allEventsIndex}`);
                  } else {
                      console.error(`DEBUG: [Timeout] FAILED to find element to highlight for index ${event.allEventsIndex}`);
                  }
              }, 50); // A small delay to ensure the re-render completes

          } else {
              console.error(`DEBUG: [playNextInQueue] CRITICAL: FAILED to advance slider. displayStep: ${displayStep}, wwOriginals: ${!!window.wwOriginals}`);
          }
      }

      let audioPath = null;
      let audioKey = null;
      if (event.audioEvent) {
          audioKey = event.audioEvent.speaker === 'moderator' ? `moderator:${event.audioEvent.message}` : `${event.audioEvent.speaker}:${event.audioEvent.message}`;
          audioPath = audioMap[audioKey];
      }

      if (audioPath) {
          console.debug(`DEBUG: [playNextInQueue] Popped event for index: ${event.allEventsIndex}. Audio key: "${audioKey}"`);
          console.debug(`DEBUG: [playNextInQueue] Playing audio: ${audioPath}`);
          audioState.audioPlayer.src = audioPath;
          audioState.audioPlayer.playbackRate = audioState.playbackRate;
          audioState.audioPlayer.onended = () => {
              console.debug(`DEBUG: [onended] Audio for index ${event.allEventsIndex} finished.`);
              audioState.isAudioPlaying = false;
              if (!audioState.isPaused && isContinuous) { 
                console.debug("DEBUG: [onended] Calling playNextInQueue recursively.");
                playNextInQueue(isContinuous);
              } else {
                console.debug("DEBUG: [onended] Loop stopped. isPaused or !isContinuous.");
              }
          };
          audioState.audioPlayer.onerror = () => {
              console.error(`DEBUG: [onerror] Audio failed to play for key: "${audioKey}"`);
              audioState.isAudioPlaying = false;
              playNextInQueue(isContinuous); 
          };
          audioState.audioPlayer.play().catch(e => {
              console.error(`DEBUG: [play.catch] Audio failed to play:`, e);
              audioState.isAudioPlaying = false;
              playNextInQueue(isContinuous);
          });
      } else {
          console.warn(`DEBUG: [playNextInQueue] No audio for event index: ${event.allEventsIndex}. Using setTimeout.`);
          setTimeout(() => {
              audioState.isAudioPlaying = false;
              if (!audioState.isPaused && isContinuous) {
                  playNextInQueue(isContinuous);
              }
          }, context.speed);
      }
  }

  function stopAndClearAudio() {
      if (audioState.isAudioPlaying) {
          audioState.audioPlayer.pause();
          audioState.isAudioPlaying = false;
      }
      audioState.audioQueue = [];
      audioState.currentlyPlayingIndex = -1;

      const currentParent = document.getElementById(parentId);
      if (currentParent) {
        // Clear any "now-playing" highlights
        const nowPlayingElement = currentParent.querySelector('#chat-log .now-playing');
        if (nowPlayingElement) {
            nowPlayingElement.classList.remove('now-playing');
        }
      }
  }

  function initThreeJs() {
    if (threeState.initialized) {
      if (threeState.demo && threeState.demo._parent && !parent.contains(threeState.demo._parent)) {
          parent.appendChild(threeState.demo._threejs.domElement);
          parent.appendChild(threeState.demo._labelRenderer.domElement);
      }
      return;
    }

    const loadAndSetup = async () => {
        try {
            const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js');
            const { OrbitControls } = await import('https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js');
            const { FBXLoader } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/FBXLoader.js');
            const { SkeletonUtils } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/utils/SkeletonUtils.js');
            const { CSS2DRenderer, CSS2DObject } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/renderers/CSS2DRenderer.js');
            const { EffectComposer } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/EffectComposer.js');
            const { RenderPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/RenderPass.js');
            const { UnrealBloomPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/UnrealBloomPass.js');
            const { ShaderPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/ShaderPass.js');
            const { FilmPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/FilmPass.js');

            class BasicWorldDemo {
              constructor(options) {
                this._Initialize(options, THREE, OrbitControls, FBXLoader, SkeletonUtils, CSS2DRenderer, CSS2DObject, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass);
              }

              _Initialize(options, THREE, OrbitControls, FBXLoader, SkeletonUtils, CSS2DRenderer, CSS2DObject, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass) {
                this._parent = options.parent;
                this._width = options.width;
                this._height = options.height;

                // WebGL Renderer with enhanced settings
                this._threejs = new THREE.WebGLRenderer({
                  antialias: true,
                  alpha: true,
                  powerPreference: "high-performance"
                });
                this._threejs.shadowMap.enabled = true;
                this._threejs.shadowMap.type = THREE.PCFSoftShadowMap;
                this._threejs.shadowMap.autoUpdate = true;
                this._threejs.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                this._threejs.setSize(this._width, this._height);
                this._threejs.outputEncoding = THREE.sRGBEncoding;
                this._threejs.toneMapping = THREE.ACESFilmicToneMapping;
                this._threejs.toneMappingExposure = 1.2;
                this._threejs.domElement.style.position = 'absolute';
                this._threejs.domElement.style.top = '0';
                this._threejs.domElement.style.left = '0';
                this._threejs.domElement.style.zIndex = '0';
                this._parent.appendChild(this._threejs.domElement);

                // CSS2D Renderer
                this._labelRenderer = new CSS2DRenderer();
                this._labelRenderer.setSize(this._width, this._height);
                this._labelRenderer.domElement.style.position = 'absolute';
                this._labelRenderer.domElement.style.top = '0px';
                this._labelRenderer.domElement.style.left = '0px';
                this._labelRenderer.domElement.style.zIndex = '1'; // On top of 3D, behind UI
                this._labelRenderer.domElement.style.pointerEvents = 'none';
                this._parent.appendChild(this._labelRenderer.domElement);

                const fov = 60;
                const aspect = this._width / this._height;
                const near = 1.0;
                const far = 100000.0;
                this._camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
                this._camera.position.set(0, 0, 50);

                this._scene = new THREE.Scene();
                this._scene.fog = new THREE.FogExp2(0x2a2a4a, 0.01); // Start with day fog color

                this._createSkybox(THREE);
                this._createAdvancedLighting(THREE);
                this._setupPostProcessing(THREE, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass);

                this._controls = new OrbitControls(this._camera, this._threejs.domElement);
                this._controls.target.set(0, 0, 0);
                this._controls.enableKeys = false;
                this._controls.update();

                this._votingArcsGroup = new THREE.Group();
                this._votingArcsGroup.name = 'votingArcs';
                this._scene.add(this._votingArcsGroup);

                this._targetRingsGroup = new THREE.Group();
                this._targetRingsGroup.name = 'targetRings';
                this._scene.add(this._targetRingsGroup);

                this._activeVoteArcs = new Map();
                this._activeTargetRings = new Map();

                this._speakingAnimations = [];

                this._LoadModels(THREE, FBXLoader, SkeletonUtils, CSS2DObject);
                this._RAF();
              }

              _createSkybox(THREE) {
                // Store THREE reference first
                this._THREE = THREE;
                
                const skyboxSize = 1000;
                const skyboxGeo = new THREE.BoxGeometry(skyboxSize, skyboxSize, skyboxSize);
                
                // Create materials for each face with initial day colors
                this._skyboxMaterials = [];
                for (let i = 0; i < 6; i++) {
                    const mat = new THREE.MeshBasicMaterial({
                        color: new THREE.Color(0x87ceeb), // Start with day color
                        side: THREE.BackSide
                    });
                    this._skyboxMaterials.push(mat);
                }

                const skybox = new THREE.Mesh(skyboxGeo, this._skyboxMaterials);
                this._skybox = skybox;
                this._scene.add(skybox);

                // Create dynamic sky canvas for the back panel (where moon/sun appears)
                const backCanvas = document.createElement('canvas');
                const canvasSize = 2048;
                backCanvas.width = canvasSize;
                backCanvas.height = canvasSize;
                this._skyCanvas = backCanvas;
                this._skyContext = backCanvas.getContext('2d');
                
                // Store celestial body properties
                this._celestialBody = {
                    x: canvasSize / 2,
                    y: canvasSize / 3,
                    size: 250,
                    phase: 0.0 // Start with day (0 = day, 1 = night)
                };

                // Create texture from canvas
                this._skyTexture = new THREE.CanvasTexture(backCanvas);
                this._skyboxMaterials[4].map = this._skyTexture;
                
                // Load moon image
                const moonImage = new Image();
                moonImage.crossOrigin = 'Anonymous';
                moonImage.onload = () => {
                    this._moonImage = moonImage;
                    this._updateSkybox(0.0); // Start with day
                };
                moonImage.onerror = () => {
                    console.error("Failed to load moon texture for skybox.");
                    this._updateSkybox(0.0); // Start with day
                };
                moonImage.src = 'assets/moon4.png';
                
                // Create stars for night sky
                this._createStars(THREE);
              }
              
              _createStars(THREE) {
                const starsGeometry = new THREE.BufferGeometry();
                const starCount = 2000;
                const positions = new Float32Array(starCount * 3);
                const colors = new Float32Array(starCount * 3);
                const sizes = new Float32Array(starCount);
                
                for (let i = 0; i < starCount; i++) {
                    const i3 = i * 3;
                    
                    // Random position on a large sphere
                    const theta = Math.random() * Math.PI * 2;
                    const phi = Math.acos(2 * Math.random() - 1);
                    const radius = 400 + Math.random() * 100;
                    
                    positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
                    positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
                    positions[i3 + 2] = radius * Math.cos(phi);
                    
                    // Star colors (white to slightly blue/yellow)
                    const starColor = new THREE.Color();
                    const colorChoice = Math.random();
                    if (colorChoice < 0.3) {
                        starColor.setHSL(0.6, 0.1, 0.9); // Bluish
                    } else if (colorChoice < 0.6) {
                        starColor.setHSL(0.1, 0.1, 0.95); // Yellowish
                    } else {
                        starColor.setHSL(0, 0, 1); // Pure white
                    }
                    colors[i3] = starColor.r;
                    colors[i3 + 1] = starColor.g;
                    colors[i3 + 2] = starColor.b;
                    
                    sizes[i] = Math.random() * 2 + 0.5;
                }
                
                starsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                starsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                starsGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
                
                const starsMaterial = new THREE.ShaderMaterial({
                    uniforms: {
                        phase: { value: 0.0 } // Start with day (0 = day, 1 = night)
                    },
                    vertexShader: `
                        attribute float size;
                        attribute vec3 color;
                        varying vec3 vColor;
                        uniform float phase;
                        
                        void main() {
                            vColor = color;
                            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                            gl_PointSize = size * (300.0 / -mvPosition.z) * phase;
                            gl_Position = projectionMatrix * mvPosition;
                        }
                    `,
                    fragmentShader: `
                        varying vec3 vColor;
                        uniform float phase;
                        
                        void main() {
                            float dist = distance(gl_PointCoord, vec2(0.5));
                            if (dist > 0.5) discard;
                            
                            float alpha = (1.0 - dist * 2.0) * phase * 0.8;
                            gl_FragColor = vec4(vColor, alpha);
                        }
                    `,
                    transparent: true,
                    blending: THREE.AdditiveBlending,
                    depthWrite: false
                });
                
                this._stars = new THREE.Points(starsGeometry, starsMaterial);
                this._starsMaterial = starsMaterial;
                this._scene.add(this._stars);
              }
              
              _updateSkybox(phase) {
                if (!this._skyContext || !this._skyCanvas) {
                    console.debug('Skybox context not ready');
                    return;
                }
                
                const ctx = this._skyContext;
                const canvas = this._skyCanvas;
                const celestial = this._celestialBody;
                
                // Clear canvas with a transparent background
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Create gradient overlay
                const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
                
                if (phase > 0.5) {
                    // Night sky
                    const nightIntensity = (phase - 0.5) * 2;
                    gradient.addColorStop(0, `rgba(10, 10, 40, ${nightIntensity})`);
                    gradient.addColorStop(0.3, `rgba(20, 20, 60, ${nightIntensity})`);
                    gradient.addColorStop(1, `rgba(5, 5, 20, ${nightIntensity})`);
                    
                    // Fill with gradient
                    ctx.fillStyle = gradient;
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw stars manually
                    ctx.fillStyle = 'white';
                    for (let i = 0; i < 200; i++) {
                        const x = Math.random() * canvas.width;
                        const y = Math.random() * canvas.height;
                        const size = Math.random() * 2;
                        ctx.globalAlpha = nightIntensity * (0.3 + Math.random() * 0.7);
                        ctx.beginPath();
                        ctx.arc(x, y, size, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    
                    // Draw moon
                    ctx.globalAlpha = nightIntensity;
                    if (this._moonImage) {
                        const moonSize = celestial.size;
                        const moonX = celestial.x;
                        const moonY = celestial.y;
                        ctx.drawImage(this._moonImage,
                            moonX - moonSize/2,
                            moonY - moonSize/2,
                            moonSize,
                            moonSize
                        );
                    } else {
                        // Fallback: draw procedural moon
                        ctx.fillStyle = '#f0f0e0';
                        ctx.shadowBlur = 50;
                        ctx.shadowColor = '#f0f0e0';
                        ctx.beginPath();
                        ctx.arc(celestial.x, celestial.y, celestial.size/2, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.shadowBlur = 0;
                    }
                } else {
                    // Day sky
                    const dayIntensity = 1 - phase * 2;
                    gradient.addColorStop(0, `rgba(135, 206, 250, ${dayIntensity})`);
                    gradient.addColorStop(0.5, `rgba(135, 206, 235, ${dayIntensity})`);
                    gradient.addColorStop(1, `rgba(255, 255, 200, ${dayIntensity * 0.5})`);
                    
                    // Fill with gradient
                    ctx.fillStyle = gradient;
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw sun
                    ctx.globalAlpha = dayIntensity;
                    
                    // Sun glow
                    const glowGradient = ctx.createRadialGradient(
                        celestial.x, celestial.y, 0,
                        celestial.x, celestial.y, celestial.size * 1.5
                    );
                    glowGradient.addColorStop(0, 'rgba(255, 255, 200, 0.8)');
                    glowGradient.addColorStop(0.3, 'rgba(255, 220, 100, 0.4)');
                    glowGradient.addColorStop(1, 'rgba(255, 200, 50, 0)');
                    
                    ctx.fillStyle = glowGradient;
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Sun core
                    ctx.fillStyle = '#ffff99';
                    ctx.shadowBlur = 80;
                    ctx.shadowColor = '#ffcc00';
                    ctx.beginPath();
                    ctx.arc(celestial.x, celestial.y, celestial.size/2, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                    
                    // Sun rays
                    ctx.strokeStyle = `rgba(255, 220, 100, ${dayIntensity * 0.5})`;
                    ctx.lineWidth = 3;
                    for (let i = 0; i < 12; i++) {
                        const angle = (i / 12) * Math.PI * 2;
                        const innerRadius = celestial.size * 0.6;
                        const outerRadius = celestial.size * 1.2;
                        ctx.beginPath();
                        ctx.moveTo(
                            celestial.x + Math.cos(angle) * innerRadius,
                            celestial.y + Math.sin(angle) * innerRadius
                        );
                        ctx.lineTo(
                            celestial.x + Math.cos(angle) * outerRadius,
                            celestial.y + Math.sin(angle) * outerRadius
                        );
                        ctx.stroke();
                    }
                }
                
                // Update texture
                if (this._skyTexture) {
                    this._skyTexture.needsUpdate = true;
                }
                
                // Update skybox colors for all faces with more dramatic changes
                if (this._skyboxMaterials && this._THREE) {
                    const THREE = this._THREE;
                    const nightColor = new THREE.Color(0x000011); // Very dark blue
                    const dayColor = new THREE.Color(0x87ceeb); // Sky blue
                    const currentColor = new THREE.Color();
                    currentColor.copy(dayColor).lerp(nightColor, phase);
                    
                    this._skyboxMaterials.forEach((mat, index) => {
                        if (index !== 4) { // Skip the back panel with moon/sun
                            mat.color.copy(currentColor);
                            mat.needsUpdate = true;
                        }
                    });
                    
                    // Force update the canvas texture
                    if (this._skyTexture) {
                        this._skyTexture.needsUpdate = true;
                    }
                }
                
                // Store current phase
                if (celestial) {
                    celestial.phase = phase;
                }
              }

              _createAdvancedLighting(THREE) {
                // Enhanced ambient lighting with color variation
                const ambientLight = new THREE.AmbientLight(0x404080, 0.4);
                ambientLight.name = 'ambientLight';
                this._scene.add(ambientLight);

                // Main directional light (moon/sun)
                const mainLight = new THREE.DirectionalLight(0xffffff, 1.8);
                mainLight.position.set(30, 50, 20);
                mainLight.castShadow = true;
                mainLight.shadow.mapSize.width = 2048;
                mainLight.shadow.mapSize.height = 2048;
                mainLight.shadow.camera.near = 0.5;
                mainLight.shadow.camera.far = 100;
                mainLight.shadow.camera.left = -50;
                mainLight.shadow.camera.right = 50;
                mainLight.shadow.camera.top = 50;
                mainLight.shadow.camera.bottom = -50;
                mainLight.shadow.bias = -0.001;
                mainLight.shadow.normalBias = 0.02;
                this._scene.add(mainLight);

                // Rim light for dramatic effect
                const rimLight = new THREE.DirectionalLight(0x8080ff, 0.8);
                rimLight.position.set(-20, 10, -30);
                this._scene.add(rimLight);

                // Atmospheric hemisphere light
                const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x1e1e3f, 0.6);
                this._scene.add(hemiLight);

                // Ground fill light
                const fillLight = new THREE.DirectionalLight(0x404080, 0.3);
                fillLight.position.set(0, -1, 0);
                this._scene.add(fillLight);

                // Store references for phase updates
                this._mainLight = mainLight;
                this._rimLight = rimLight;
                this._hemiLight = hemiLight;
                this._fillLight = fillLight;

                // Create a spotlight for night actions
                const spotLight = new THREE.SpotLight(0xffffff, 5.0, 50, Math.PI / 4, 0.5, 2);
                spotLight.position.set(0, 25, 0);
                spotLight.castShadow = true;
                spotLight.visible = false;
                this._scene.add(spotLight);
                this._spotLight = spotLight;
                this._scene.add(spotLight.target);
              }

              _createMysticalCircles(THREE, radius) {
                // Create multiple concentric circles with mystical patterns
                for (let i = 0; i < 3; i++) {
                  const circleRadius = radius - (i * 2) - 1;
                  const circleGeometry = new THREE.RingGeometry(circleRadius - 0.1, circleRadius + 0.1, 64);
                  const circleMaterial = new THREE.MeshStandardMaterial({
                    color: new THREE.Color().setHSL(0.6 + i * 0.1, 0.8, 0.3 + i * 0.1),
                    emissive: new THREE.Color().setHSL(0.6 + i * 0.1, 0.5, 0.1),
                    emissiveIntensity: 0.2,
                    transparent: true,
                    opacity: 0.6 - i * 0.1,
                    side: THREE.DoubleSide
                  });
                  
                  const circle = new THREE.Mesh(circleGeometry, circleMaterial);
                  circle.rotation.x = -Math.PI / 2;
                  circle.position.y = 0.01 + i * 0.001;
                  this._scene.add(circle);
                }

                // Add runic symbols around the outer circle
                // this._createRunicSymbols(THREE, radius);
              }

              _createRunicSymbols(THREE, radius) {
                const symbolCount = 8;
                const symbolGeometry = new THREE.PlaneGeometry(1, 1);
                
                for (let i = 0; i < symbolCount; i++) {
                  const angle = (i / symbolCount) * Math.PI * 2;
                  const x = (radius + 3) * Math.sin(angle);
                  const z = (radius + 3) * Math.cos(angle);
                  
                  // Create a simple runic-like pattern using canvas
                  const canvas = document.createElement('canvas');
                  canvas.width = 64;
                  canvas.height = 64;
                  const ctx = canvas.getContext('2d');
                  
                  ctx.fillStyle = 'rgba(100, 150, 255, 0.8)';
                  ctx.font = '40px serif';
                  ctx.textAlign = 'center';
                  ctx.fillText(['ᚠ', 'ᚡ', 'ᚢ', 'ᚣ', 'ᚤ', 'ᚥ', 'ᚦ', 'ᚧ'][i], 32, 45);
                  
                  const symbolMaterial = new THREE.MeshBasicMaterial({
                    map: new THREE.CanvasTexture(canvas),
                    transparent: true,
                    alphaTest: 0.1
                  });
                  
                  const symbol = new THREE.Mesh(symbolGeometry, symbolMaterial);
                  symbol.position.set(x, 0.15, z);
                  symbol.rotation.x = -Math.PI / 2;
                  this._scene.add(symbol);
                }
              }

              _createParticleSystem(THREE) {
                // Create floating mystical particles
                const particleCount = 150;
                const particles = new THREE.BufferGeometry();
                const positions = new Float32Array(particleCount * 3);
                const colors = new Float32Array(particleCount * 3);
                const sizes = new Float32Array(particleCount);
                
                for (let i = 0; i < particleCount; i++) {
                  const i3 = i * 3;
                  
                  // Random position within a larger area
                  positions[i3] = (Math.random() - 0.5) * 80;
                  positions[i3 + 1] = Math.random() * 30 + 5;
                  positions[i3 + 2] = (Math.random() - 0.5) * 80;
                  
                  // Mystical colors (blues, purples, greens)
                  const hue = Math.random() * 0.3 + 0.5; // 0.5-0.8 range
                  const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
                  colors[i3] = color.r;
                  colors[i3 + 1] = color.g;
                  colors[i3 + 2] = color.b;
                  
                  sizes[i] = Math.random() * 2 + 0.5;
                }
                
                particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                particles.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                particles.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
                
                const particleMaterial = new THREE.ShaderMaterial({
                  uniforms: {
                    time: { value: 0 }
                  },
                  vertexShader: `
                    attribute float size;
                    attribute vec3 color;
                    varying vec3 vColor;
                    uniform float time;
                    
                    void main() {
                      vColor = color;
                      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                      gl_PointSize = size * (300.0 / -mvPosition.z) * (1.0 + sin(time * 2.0 + position.x * 0.1) * 0.3);
                      gl_Position = projectionMatrix * mvPosition;
                    }
                  `,
                  fragmentShader: `
                    varying vec3 vColor;
                    
                    void main() {
                      float dist = distance(gl_PointCoord, vec2(0.5));
                      if (dist > 0.5) discard;
                      
                      float alpha = 1.0 - (dist * 2.0);
                      alpha *= alpha; // Softer edges
                      
                      gl_FragColor = vec4(vColor, alpha * 0.6);
                    }
                  `,
                  transparent: true,
                  blending: THREE.AdditiveBlending,
                  depthWrite: false
                });
                
                this._particles = new THREE.Points(particles, particleMaterial);
                this._scene.add(this._particles);
                this._particleMaterial = particleMaterial;
              }

              _setupPostProcessing(THREE, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass) {
                // Create effect composer
                this._composer = new EffectComposer(this._threejs);
                
                // Render pass
                const renderPass = new RenderPass(this._scene, this._camera);
                this._composer.addPass(renderPass);

                // Bloom pass for glowing effects - balanced for day
                const bloomPass = new UnrealBloomPass(
                  new THREE.Vector2(this._width, this._height),
                  0.15,  // strength - moderate for day
                  0.333,   // radius - good coverage for day
                  0.4    // threshold - balanced to allow some bloom
                );
                this._composer.addPass(bloomPass);

                // Film grain for atmosphere
                const filmPass = new FilmPass(
                  0.15,  // noise intensity
                  0.1,   // scanline intensity
                  0,     // scanline count
                  false  // grayscale
                );
                this._composer.addPass(filmPass);

                // Custom atmospheric shader
                const atmosphereShader = {
                  uniforms: {
                    'tDiffuse': { value: null },
                    'time': { value: 0.0 },
                    'phase': { value: 0.0 } // 0 = day, 1 = night
                  },
                  vertexShader: `
                    varying vec2 vUv;
                    void main() {
                      vUv = uv;
                      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                  `,
                  fragmentShader: `
                    uniform sampler2D tDiffuse;
                    uniform float time;
                    uniform float phase;
                    varying vec2 vUv;
                    
                    void main() {
                      vec4 color = texture2D(tDiffuse, vUv);
                      
                      // Add subtle color grading based on phase
                      if (phase > 0.5) {
                        // Night - add blue tint and increase contrast
                        color.rgb = mix(color.rgb, color.rgb * vec3(0.8, 0.9, 1.2), 0.3);
                        color.rgb = pow(color.rgb, vec3(1.1));
                      } else {
                        // Day - add warm tint
                        color.rgb = mix(color.rgb, color.rgb * vec3(1.1, 1.05, 0.95), 0.2);
                      }
                      
                      // Add subtle vignette
                      vec2 center = vec2(0.5, 0.5);
                      float dist = distance(vUv, center);
                      float vignette = 1.0 - smoothstep(0.3, 0.8, dist);
                      color.rgb *= mix(0.7, 1.0, vignette);
                      
                      gl_FragColor = color;
                    }
                  `
                };

                this._atmospherePass = new ShaderPass(atmosphereShader);
                this._composer.addPass(this._atmospherePass);

                // Store references
                this._bloomPass = bloomPass;
                this._filmPass = filmPass;
              }

              _LoadModels(THREE, FBXLoader, SkeletonUtils, CSS2DObject) {
                this._playerObjects = new Map();
                this._playerGroup = new THREE.Group();
                this._playerGroup.name = 'playerGroup';

                // Create enhanced ground circle with better materials
                const radius = 15;
                const groundGeometry = new THREE.CircleGeometry(radius + 5, 64);
                const groundMaterial = new THREE.MeshStandardMaterial({
                    color: 0x1a1a2a,
                    roughness: 0.9,
                    metalness: 0.1,
                    normalScale: new THREE.Vector2(0.5, 0.5),
                    transparent: true,
                    opacity: 0.95
                });
                
                // Add a subtle normal map pattern
                const canvas = document.createElement('canvas');
                canvas.width = 512;
                canvas.height = 512;
                const ctx = canvas.getContext('2d');
                const imageData = ctx.createImageData(512, 512);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    const noise = Math.random() * 0.1 + 0.5;
                    imageData.data[i] = Math.floor(noise * 255);     // R
                    imageData.data[i + 1] = Math.floor(noise * 255); // G
                    imageData.data[i + 2] = 255;                     // B
                    imageData.data[i + 3] = 255;                     // A
                }
                ctx.putImageData(imageData, 0, 0);
                groundMaterial.normalMap = new THREE.CanvasTexture(canvas);
                
                const ground = new THREE.Mesh(groundGeometry, groundMaterial);
                ground.rotation.x = -Math.PI / 2;
                ground.position.y = -0.1;
                ground.receiveShadow = true;
                this._scene.add(ground);

                // Add mystical circle patterns
                this._createMysticalCircles(THREE, radius);

                this._scene.add(this._playerGroup);

                // Store references for later use
                this._THREE = THREE;
                this._CSS2DObject = CSS2DObject;
                
                // Create particle system for atmosphere
                this._createParticleSystem(THREE);
                
                // Frame the empty group initially with better camera positioning
                this._camera.position.set(25, 30, 35);
                this._controls.target.set(0, 8, 0);
                this._controls.enableDamping = true;
                this._controls.dampingFactor = 0.05;
                this._controls.minDistance = 20;
                this._controls.maxDistance = 80;
                this._controls.maxPolarAngle = Math.PI * 0.75;
                this._controls.update();
              }

              focusOnPlayer(playerName, leftPanelWidth = 0, rightPanelWidth = 0) {
                if (!this._playerGroup || this._playerGroup.children.length === 0 || !this._THREE || !this._playerObjects) {
                    return;
                }
                const player = this._playerObjects.get(playerName);
                if (!player) return;

                // --- 1. Calculate the required camera distance ---
                
                // First, determine the real viewport size, excluding the UI panels
                const effectiveWidth = this._width - leftPanelWidth - rightPanelWidth;
                const effectiveHeight = this._height;
                
                // Get the bounding box of the entire group of players
                const viewBox = new this._THREE.Box3().setFromObject(this._playerGroup);
                const viewSize = viewBox.getSize(new this._THREE.Vector3());
                const viewCenter = viewBox.getCenter(new this._THREE.Vector3());

                // Calculate the camera's field of view in radians
                const fov = this._camera.fov * (Math.PI / 180);
                const aspect = effectiveWidth / effectiveHeight;
                
                // Derive the horizontal FoV from the vertical FoV and the new aspect ratio
                const horizontalFov = 2 * Math.atan(Math.tan(fov / 2) * aspect);
                
                // Calculate the distance needed to fit the content vertically and horizontally
                const distV = (viewSize.y / 2) / Math.tan(fov / 2);
                const distH = (viewSize.x / 2) / Math.tan(horizontalFov / 2);
                
                // The required distance is the larger of the two, plus some padding
                let distance = Math.max(distV, distH) * 1.05;
                
                // --- 2. Position the camera using the calculated distance ---

                const playerPosition = player.container.position.clone();
                const direction = playerPosition.clone().normalize();
                
                // We preserve the angle you liked by scaling the position based on the new distance.
                // The camera is positioned on the line extending from the center through the player.
                const endPos = playerPosition.clone().add(direction.multiplyScalar(distance * 0.6));
                endPos.y = playerPosition.y + distance * 0.5; // Elevate based on distance

                // The target remains the center of the action
                const endTarget = viewCenter;
                
                // --- 3. Animate the transition ---

                this._cameraAnimation = {
                    startTime: performance.now(),
                    duration: 1200,
                    startPos: this._camera.position.clone(),
                    endPos: endPos,
                    startTarget: this._controls.target.clone(),
                    endTarget: endTarget,
                    ease: t => 1 - Math.pow(1 - t, 3)
                };
              }

              resetCameraView() {
                if (!this._playerGroup || this._playerGroup.children.length === 0 || !this._THREE) {
                    return; // Can't frame an empty group
                }

                // Calculate the bounding box that contains all players
                const box = new this._THREE.Box3().setFromObject(this._playerGroup);
                const size = box.getSize(new this._THREE.Vector3());
                const center = box.getCenter(new this._THREE.Vector3());

                // Determine the maximum dimension of the box
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = this._camera.fov * (Math.PI / 180);
                
                // Calculate the distance the camera needs to be to fit the box
                let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                
                // Add some padding so the players aren't right at the edge of the screen
                // cameraZ *= 1.4; 
                cameraZ *= 1.1;

                // Set a nice isometric-style camera position
                const endPos = new this._THREE.Vector3(
                    center.x,
                    center.y + cameraZ / 2, // Elevate the camera
                    center.z + cameraZ       // Pull it back
                );

                // The target is the center of the player group
                const endTarget = center;

                // Use the same animation system as focusOnPlayer
                this._cameraAnimation = {
                    startTime: performance.now(),
                    duration: 1200,
                    startPos: this._camera.position.clone(),
                    endPos: endPos,
                    startTarget: this._controls.target.clone(),
                    endTarget: endTarget,
                    ease: t => 1 - Math.pow(1 - t, 3)
                };
              }

              updatePlayerActive(playerName) {
                const player = this._playerObjects.get(playerName);
                if (!player) return;
                const { orb, orbLight, body, head, shoulders, glow, pedestal, container } = player;
                
                orb.material.emissiveIntensity = 1.;
                orbLight.intensity = 1.;
                glow.material.emissiveIntensity = 0.5;
                // Slight scale up animation
                container.scale.setScalar(1.1);
                pedestal.material.emissiveIntensity = 0.3;
              }

              updatePlayerStatus(playerName, status, threatLevel = 0, is_active = false) {
                const player = this._playerObjects.get(playerName);
                if (!player) return;

                const { orb, orbLight, body, head, shoulders, glow, pedestal, container } = player;

                orb.material.color.setHex(0x00ff00);
                orb.material.emissive.setHex(0x00ff00);
                orb.material.emissiveIntensity = 0.8;
                orb.material.opacity = 0.9;
                orb.visible = true;
                orbLight.color.setHex(0x00ff00);
                orbLight.intensity = 0.8;
                orbLight.visible = true;
                body.material.color.setHex(0x4466ff);
                body.material.emissive.setHex(0x111166);
                body.material.emissiveIntensity = 0.2;
                shoulders.material.color.setHex(0x4466ff);
                shoulders.material.emissive.setHex(0x111166);
                shoulders.material.emissiveIntensity = 0.2;
                head.material.color.setHex(0xfdbcb4);
                head.material.emissive.setHex(0x442211);
                head.material.emissiveIntensity = 0.1;
                glow.material.color.setHex(0x00ff00);
                glow.material.emissive.setHex(0x00ff00);
                glow.material.emissiveIntensity = 0.3;
                glow.visible = true;
                pedestal.material.emissive.setHex(0x111122);
                pedestal.material.emissiveIntensity = 0.1;
                container.scale.setScalar(1.0);
                container.position.y = 0;
                container.rotation.x = 0;
                if (player.nameplate && player.nameplate.element) {
                    player.nameplate.element.style.transition = 'opacity 0.5s ease-in';
                    player.nameplate.element.style.opacity = '1.0';
                }
                player.isAlive = true;
                
                switch(status) {
                    case 'dead':
                        orb.visible = false;
                        orbLight.visible = false;
                        glow.visible = false;
                        body.material.color.setHex(0x444444);
                        body.material.emissive.setHex(0x000000);
                        shoulders.material.color.setHex(0x444444);
                        shoulders.material.emissive.setHex(0x000000);
                        head.material.color.setHex(0x666666);
                        head.material.emissive.setHex(0x000000);
                        pedestal.material.emissive.setHex(0x000000);
                        // Sink into ground
                        container.position.y = -1.5;
                        // Tilt slightly
                        container.rotation.x = 0.2;
                        // Fade out nameplate
                        if (player.nameplate && player.nameplate.element) {
                            player.nameplate.element.style.transition = 'opacity 2s ease-out';
                            player.nameplate.element.style.opacity = '0.2';
                        }
                        player.isAlive = false;
                        break;
                    case 'werewolf':
                        body.material.color.setHex(0x880000);
                        body.material.emissive.setHex(0x440000);
                        body.material.emissiveIntensity = 0.3;
                        shoulders.material.color.setHex(0x880000);
                        shoulders.material.emissive.setHex(0x440000);
                        shoulders.material.emissiveIntensity = 0.3;
                        glow.material.color.setHex(0xff0000);
                        glow.material.emissive.setHex(0xff0000);
                        glow.material.emissiveIntensity = 0.4;
                        glow.visible = true;
                        pedestal.material.emissive.setHex(0x440000);
                        pedestal.material.emissiveIntensity = 0.2;
                        break;
                    case 'doctor':
                        body.material.color.setHex(0x008800);
                        body.material.emissive.setHex(0x004400);
                        body.material.emissiveIntensity = 0.3;
                        shoulders.material.color.setHex(0x008800);
                        shoulders.material.emissive.setHex(0x004400);
                        shoulders.material.emissiveIntensity = 0.3;
                        glow.material.color.setHex(0x00ff00);
                        glow.material.emissive.setHex(0x00ff00);
                        glow.material.emissiveIntensity = 0.4;
                        glow.visible = true;
                        pedestal.material.emissive.setHex(0x004400);
                        pedestal.material.emissiveIntensity = 0.2;
                        break;
                    case 'seer':
                        body.material.color.setHex(0x4B0082);
                        body.material.emissive.setHex(0x3A005A);
                        body.material.emissiveIntensity = 0.3;
                        shoulders.material.color.setHex(0x4B0082);
                        shoulders.material.emissive.setHex(0x3A005A);
                        shoulders.material.emissiveIntensity = 0.3;
                        glow.material.color.setHex(0x9932CC);
                        glow.material.emissive.setHex(0x9932CC);
                        glow.material.emissiveIntensity = 0.4;
                        glow.visible = true;
                        pedestal.material.emissive.setHex(0x3A005A);
                        pedestal.material.emissiveIntensity = 0.2;
                        break;
                    default:
                        // This is now covered by the reset block at the top of the function.
                        break;
                }

                if (threatLevel >= 1.0) { // DANGER
                    orb.material.color.setHex(0xff0000); // Red
                    orb.material.emissive.setHex(0xff0000);
                    orb.material.emissiveIntensity = 1.0;
                    orb.material.opacity = 0.9;
                    orbLight.color.setHex(0xff0000);
                    orbLight.intensity = 1.2;
                    glow.material.color.setHex(0xff0000);
                    glow.material.emissive.setHex(0xff0000);
                    glow.material.emissiveIntensity = 0.3;
                } else if (threatLevel >= 0.5) { // UNEASY
                    orb.material.color.setHex(0xffff00); // Yellow
                    orb.material.emissive.setHex(0xffff00);
                    orb.material.emissiveIntensity = 1.0;
                    orb.material.opacity = 0.9;
                    orbLight.color.setHex(0xffff00);
                    orbLight.intensity = 1.2;
                    glow.material.color.setHex(0xffff00);
                    glow.material.emissive.setHex(0xffff00);
                    glow.material.emissiveIntensity = 0.3;
                } else { // SAFE
                    // orb.material.color.setHex(0x00ff00); // Green
                    orb.material.color.setHex(0x00ff00);
                    orb.material.emissive.setHex(0x00ff00);
                    orb.material.emissiveIntensity = 1.0;
                    orb.material.opacity = 0.9;
                    orbLight.color.setHex(0x00ff00);
                    orbLight.intensity = 1.2;
                    glow.material.color.setHex(0x00ff00);
                    glow.material.emissive.setHex(0x00ff00);
                    glow.material.emissiveIntensity = 0.3;
                }
              }

              triggerSpeakingAnimation(playerName) {
                const player = this._playerObjects.get(playerName);
                if (!player || !player.isAlive) return;

                const wave = this._createSoundWave(this._THREE);
                player.container.add(wave);

                // Add the wave to our animation manager array
                this._speakingAnimations.push({
                    mesh: wave,
                    startTime: performance.now(),
                    duration: 1800, // Animation duration in milliseconds
                });
              }

              _createSoundWave(THREE) {
                const waveGeometry = new THREE.RingGeometry(0.5, 0.7, 32);
                const waveMaterial = new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    transparent: true,
                    opacity: 0.8,
                    side: THREE.DoubleSide,
                });
                const wave = new THREE.Mesh(waveGeometry, waveMaterial);

                // Position the wave horizontally at the player's feet
                wave.rotation.x = -Math.PI / 2;
                wave.position.y = 0.25; // Slightly above the pedestal
                return wave;
              }

              _createVoteParticleTrail(voterName, targetName, color = 0x00ffff) {
                const voter = this._playerObjects.get(voterName);
                const target = this._playerObjects.get(targetName);
                if (!voter || !target) return;

                const startPos = voter.container.position.clone();
                startPos.y += 1.5; // Start above the voter's head
                const endPos = target.container.position.clone();
                endPos.y += 1.5; // End above the target's head

                const midPos = new this._THREE.Vector3().addVectors(startPos, endPos).multiplyScalar(0.5);
                const dist = startPos.distanceTo(endPos);
                midPos.y += dist * 0.3; // Arc height

                const curve = new this._THREE.CatmullRomCurve3([startPos, midPos, endPos]);
                const particleCount = 50;
                const particleGeometry = new this._THREE.BufferGeometry();
                const positions = new Float32Array(particleCount * 3);
                particleGeometry.setAttribute('position', new this._THREE.BufferAttribute(positions, 3));

                const particleMaterial = new this._THREE.PointsMaterial({
                    color: color,
                    size: 0.3,
                    transparent: true,
                    opacity: 0.8,
                    blending: this._THREE.AdditiveBlending,
                    sizeAttenuation: true,
                });

                const particles = new this._THREE.Points(particleGeometry, particleMaterial);
                this._votingArcsGroup.add(particles);

                const trail = {
                    particles,
                    curve,
                    target: targetName,
                    startTime: Date.now(),
                    update: () => {
                        const elapsedTime = (Date.now() - trail.startTime) / 1000;
                        const positions = trail.particles.geometry.attributes.position.array;
                        for (let i = 0; i < particleCount; i++) {
                            const t = (elapsedTime * 0.2 + (i / particleCount)) % 1;
                            const pos = trail.curve.getPointAt(t);
                            positions[i * 3] = pos.x;
                            positions[i * 3 + 1] = pos.y;
                            positions[i * 3 + 2] = pos.z;
                        }
                        trail.particles.geometry.attributes.position.needsUpdate = true;
                    }
                };
                this._activeVoteArcs.set(voterName, trail);

                // Also add to a separate list for animation updates
                if (!this._animatingTrails) this._animatingTrails = [];
                this._animatingTrails.push(trail);
              }

              _updateTargetRing(targetName, voteCount) {
                  const target = this._playerObjects.get(targetName);
                  if (!target) return;

                  let ringData = this._activeTargetRings.get(targetName);

                  if (voteCount > 0 && !ringData) {
                      const geometry = new this._THREE.RingGeometry(2, 2.2, 32);
                      const material = new this._THREE.MeshBasicMaterial({
                          color: 0x00ffff,
                          transparent: true,
                          opacity: 0, // Start invisible, fade in
                          side: this._THREE.DoubleSide,
                      });
                      const ring = new this._THREE.Mesh(geometry, material);
                      ring.position.copy(target.container.position);
                      ring.position.y = 0.1;
                      ring.rotation.x = -Math.PI / 2;

                      this._targetRingsGroup.add(ring);
                      ringData = { ring, material, targetOpacity: 0 };
                      this._activeTargetRings.set(targetName, ringData);
                  }

                  if (ringData) {
                      if (voteCount > 0) {
                          ringData.targetOpacity = 0.3 + Math.min(voteCount * 0.2, 0.7);
                      } else {
                          ringData.targetOpacity = 0;
                      }
                  }
              }

              updateVoteVisuals(votes, clearAll = false) {
                  if (!this._playerObjects || this._playerObjects.size === 0) return;

                  if (clearAll) {
                      votes.clear();
                  }

                  // Remove arcs from players who are no longer voting or if clearing all
                  this._activeVoteArcs.forEach((trail, voterName) => {
                      if (!votes.has(voterName)) {
                          this._votingArcsGroup.remove(trail.particles);
                          this._activeVoteArcs.delete(voterName);
                          if (this._animatingTrails) {
                              this._animatingTrails = this._animatingTrails.filter(t => t !== trail);
                          }
                      }
                  });


                  // Update existing arcs or create new ones
                  votes.forEach((voteData, voterName) => {
                      const { target: targetName, type } = voteData;
                      const existingTrail = this._activeVoteArcs.get(voterName);

                      let color = 0x00ffff; // Default to cyan
                      if (type === 'night_vote') color = 0xff0000; // Red
                      else if (type === 'doctor_heal_action') color = 0x00ff00; // Green
                      else if (type === 'seer_inspection') color = 0x800080; // Purple

                      if (existingTrail) {
                          if (existingTrail.target !== targetName) {
                              this._votingArcsGroup.remove(existingTrail.particles);
                               if (this._animatingTrails) {
                                  this._animatingTrails = this._animatingTrails.filter(t => t !== existingTrail);
                              }
                              this._createVoteParticleTrail(voterName, targetName, color);
                          }
                      } else {
                          this._createVoteParticleTrail(voterName, targetName, color);
                      }
                  });

                  // Update target rings
                  const targetVoteCounts = new Map();
                  votes.forEach((voteData) => {
                      const { target: targetName } = voteData;
                      targetVoteCounts.set(targetName, (targetVoteCounts.get(targetName) || 0) + 1);
                  });

                  this._playerObjects.forEach((player, playerName) => {
                      this._updateTargetRing(playerName, targetVoteCounts.get(playerName) || 0);
                  });
              }

              updatePhase(phase) {
                if (!this._scene) return;
                
                // Handle various phase formats (DAY, NIGHT, or lowercase)
                const normalizedPhase = (phase || 'DAY').toUpperCase();
                
                // Calculate target phase value (0 = day, 1 = night)
                const targetPhase = normalizedPhase === 'NIGHT' ? 1.0 : 0.0;
                
                // Initialize transition system if not exists
                if (!this._phaseTransition) {
                    this._phaseTransition = {
                        current: targetPhase,
                        target: targetPhase,
                        speed: 0.05 // Increased transition speed for testing
                    };
                    // Immediately set to target on first call
                    this._updateSceneForPhase(targetPhase);
                } else if (this._phaseTransition.target !== targetPhase) {
                    // Only update if phase actually changed
                    this._phaseTransition.target = targetPhase;
                }
              }
              
              _updateSceneForPhase(phaseValue) {
                const THREE = this._THREE;
                
                // Update renderer tone mapping for day/night mood
                if (this._threejs) {
                    this._threejs.toneMappingExposure = 1.2 - phaseValue * 0.5; // Darker at night
                }
                
                // Smoothly interpolate lighting
                if (this._mainLight) {
                    const nightColor = new THREE.Color(0x6666cc); // More blue at night
                    const dayColor = new THREE.Color(0xffffcc);
                    this._mainLight.color.copy(dayColor).lerp(nightColor, phaseValue);
                    this._mainLight.intensity = 1.8 - phaseValue * 1.0; // Much dimmer at night
                    
                    // Animate light position for sun/moon movement
                    const angle = phaseValue * Math.PI * 0.3;
                    this._mainLight.position.set(
                        30 * Math.cos(angle),
                        50 - phaseValue * 20,
                        20 * Math.sin(angle)
                    );
                }
                
                if (this._rimLight) {
                    const nightColor = new THREE.Color(0x6666ff);
                    const dayColor = new THREE.Color(0xffcc99);
                    this._rimLight.color.copy(dayColor).lerp(nightColor, phaseValue);
                    this._rimLight.intensity = 0.6 + phaseValue * 0.4;
                }
                
                if (this._hemiLight) {
                    const nightSkyColor = new THREE.Color(0x4a4a6a);
                    const daySkyColor = new THREE.Color(0x87ceeb);
                    const nightGroundColor = new THREE.Color(0x1e1e3f);
                    const dayGroundColor = new THREE.Color(0x8b7355);
                    
                    this._hemiLight.color.copy(daySkyColor).lerp(nightSkyColor, phaseValue);
                    this._hemiLight.groundColor.copy(dayGroundColor).lerp(nightGroundColor, phaseValue);
                    this._hemiLight.intensity = 0.8 - phaseValue * 0.4;
                }
                
                // Update ambient light
                const ambientLight = this._scene.getObjectByName('ambientLight');
                if (ambientLight) {
                    const nightColor = new THREE.Color(0x404080);
                    const dayColor = new THREE.Color(0x606090);
                    ambientLight.color.copy(dayColor).lerp(nightColor, phaseValue);
                    ambientLight.intensity = 0.4 + phaseValue * 0.1;
                }
                
                // Smoothly transition fog - more dramatic change
                if (this._scene.fog) {
                    const nightFogColor = new THREE.Color(0x050515); // Very dark blue at night
                    const dayFogColor = new THREE.Color(0x2a2a4a); // Lighter blue during day
                    this._scene.fog.color.copy(dayFogColor).lerp(nightFogColor, phaseValue);
                    this._scene.fog.density = 0.01 + phaseValue * 0.015; // Denser fog at night
                }
                
                // Update skybox
                this._updateSkybox(phaseValue);
                
                // Update stars visibility
                if (this._starsMaterial) {
                    this._starsMaterial.uniforms.phase.value = phaseValue;
                }
                
                // Update atmosphere shader
                if (this._atmospherePass) {
                    this._atmospherePass.uniforms.phase.value = phaseValue;
                }
                
                // Update particle colors based on phase
                if (this._particles && this._particles.geometry.attributes.color) {
                    const colors = this._particles.geometry.attributes.color.array;
                    for (let i = 0; i < colors.length; i += 3) {
                        // Shift particle hue based on phase
                        const baseHue = 0.5 + Math.random() * 0.3; // Base blue-purple range
                        const phaseShift = phaseValue * 0.1; // Shift towards purple at night
                        const hue = baseHue + phaseShift;
                        const saturation = 0.8 - phaseValue * 0.2; // Less saturated at night
                        const lightness = 0.6 - phaseValue * 0.2; // Darker at night
                        
                        const color = new THREE.Color().setHSL(hue, saturation, lightness);
                        colors[i] = color.r;
                        colors[i + 1] = color.g;
                        colors[i + 2] = color.b;
                    }
                    this._particles.geometry.attributes.color.needsUpdate = true;
                }
                
                // Update bloom intensity based on phase - moderate during day, more at night
                if (this._bloomPass) {
                    this._bloomPass.strength = 0.35 + phaseValue * 0.35; // Moderate bloom during day (0.35), stronger at night (0.7)
                    this._bloomPass.radius = 0.6 + phaseValue * 0.3; // Good radius during day (0.6), wider at night (0.9)
                    this._bloomPass.threshold = 0.4 - phaseValue * 0.15; // Balanced threshold
                }
              }

              _createNameplate(name, displayName, imageUrl, CSS2DObject) {
                const container = document.createElement('div');
                container.style.backgroundColor = 'rgba(255, 255, 255, 0)';
                container.style.padding = '6px 10px';  // Slightly smaller padding
                container.style.borderRadius = '8px';
                container.style.display = 'flex';
                container.style.alignItems = 'center';
                container.style.justifyContent = 'center';
                container.style.gap = '8px';  // Reduced gap
                container.style.textAlign = 'center';

                const img = document.createElement('img');
                img.src = imageUrl;
                img.style.width = '40px';  // Reduced from 60px
                img.style.height = '40px'; // Reduced from 60px
                img.style.borderRadius = '50%';
                img.style.objectFit = 'cover';
                img.style.backgroundColor = 'white';
                img.style.border = '2px solid rgba(255, 255, 255, 0.3)';

                const textContainer = document.createElement('div');
                textContainer.style.display = 'flex';
                textContainer.style.flexDirection = 'column';
                textContainer.style.alignItems = 'center';

                const nameText = document.createElement('div');
                nameText.textContent = name;
                nameText.style.color = 'white';
                nameText.style.fontFamily = 'Arial, sans-serif';
                nameText.style.fontSize = '14px';
                nameText.style.fontWeight = '500';
                textContainer.appendChild(nameText);

                if (displayName && displayName !== "" && displayName !== name) {
                    const displayNameText = document.createElement('div');
                    displayNameText.textContent = displayName;
                    displayNameText.style.color = 'grey';
                    displayNameText.style.fontSize = '12px';
                    displayNameText.style.fontFamily = 'Arial, sans-serif';
                    displayNameText.style.marginTop = '4px';
                    textContainer.appendChild(displayNameText);
                }

                container.appendChild(img);
                container.appendChild(textContainer);

                const label = new CSS2DObject(container);
                return label;
              }

              _FrameGroup(group, THREE) {
                const box = new THREE.Box3().setFromObject(group);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = this._camera.fov * (Math.PI / 180);

                let cameraZ = Math.abs(maxDim / Math.tan(fov / 2));
                cameraZ *= 0.5;

                this._camera.position.set(center.x, center.y, center.z - cameraZ);

                const shiftY = size.y / 2.5;
                this._camera.position.y += shiftY;

                const newTarget = center.clone();
                newTarget.y += shiftY;
                this._controls.target.copy(newTarget);
                this._controls.update();
              }

              _NormalizeModel(model, THREE) {
                const box = new THREE.Box3().setFromObject(model);
                const size = box.getSize(new THREE.Vector3());
                const center = box.getCenter(new THREE.Vector3());
                model.position.copy(center).negate();
                const wrapper = new THREE.Group();
                wrapper.add(model);
                const scale = 1.0 / size.y;
                wrapper.scale.set(scale, scale, scale);
                return wrapper;
              }

              _RAF() {
                requestAnimationFrame((time) => {
                  // Animate phase transition with visual feedback
                  if (this._phaseTransition) {
                    const diff = this._phaseTransition.target - this._phaseTransition.current;
                    if (Math.abs(diff) > 0.001) {
                      this._phaseTransition.current += diff * this._phaseTransition.speed;
                      this._updateSceneForPhase(this._phaseTransition.current);
                    }
                  }
                  
                  // Update time-based uniforms
                  if (this._particleMaterial) {
                    this._particleMaterial.uniforms.time.value = time * 0.001;
                  }
                  if (this._atmospherePass) {
                    this._atmospherePass.uniforms.time.value = time * 0.001;
                  }
                  
                  // Animate particle system with phase-aware movement
                  if (this._particles) {
                    const phaseValue = this._phaseTransition ? this._phaseTransition.current : 0;
                    // Slower rotation at night
                    this._particles.rotation.y = time * 0.0001 * (1 - phaseValue * 0.5);
                    
                    const positions = this._particles.geometry.attributes.position.array;
                    for (let i = 0; i < positions.length; i += 3) {
                      // More gentle movement at night
                      const movementScale = 1 - phaseValue * 0.5;
                      positions[i + 1] += Math.sin(time * 0.001 + positions[i] * 0.01) * 0.02 * movementScale;
                      // Wrap around if particles fall too low
                      if (positions[i + 1] < 0) {
                        positions[i + 1] = 35;
                      }
                    }
                    this._particles.geometry.attributes.position.needsUpdate = true;
                  }
                  
                  // Animate stars twinkling
                  if (this._stars && this._phaseTransition && this._phaseTransition.current > 0.5) {
                    const sizes = this._stars.geometry.attributes.size.array;
                    for (let i = 0; i < sizes.length; i++) {
                      sizes[i] = (Math.random() * 2 + 0.5) * (0.8 + Math.sin(time * 0.001 + i) * 0.2);
                    }
                    this._stars.geometry.attributes.size.needsUpdate = true;
                  }

                  // Use performance.now() for more precise animation timing
                  const now = performance.now();

                  if (this._cameraAnimation) {
                    const anim = this._cameraAnimation;
                    const elapsed = now - anim.startTime;
                    let progress = Math.min(elapsed / anim.duration, 1.0);
                    
                    // Apply easing function
                    const easedProgress = anim.ease(progress);

                    // Interpolate camera position and controls target
                    this._camera.position.lerpVectors(anim.startPos, anim.endPos, easedProgress);
                    this._controls.target.lerpVectors(anim.startTarget, anim.endTarget, easedProgress);
                    this._controls.update();

                    // If animation is complete, clear it
                    if (progress >= 1.0) {
                        this._cameraAnimation = null;
                    }
                  }

                  // Animate speaking sound waves
                  this._speakingAnimations = this._speakingAnimations.filter(anim => {
                    const elapsedTime = now - anim.startTime;
                    if (elapsedTime >= anim.duration) {
                    // Animation is over, remove the mesh from the scene
                    if (anim.mesh.parent) {
                        anim.mesh.parent.remove(anim.mesh);
                    }
                    // Clean up Three.js objects to free memory
                    anim.mesh.geometry.dispose();
                    anim.mesh.material.dispose();
                    return false; // Remove from the animations array
                    }

                    // Calculate animation progress (from 0.0 to 1.0)
                    const progress = elapsedTime / anim.duration;

                    // Make the wave expand and fade out
                    anim.mesh.scale.setScalar(1 + progress * 5);
                    anim.mesh.material.opacity = 0.8 * (1 - progress);

                    return true; // Keep the animation in the array
                  });
                  
                  // Animate player objects with enhanced effects
                  if (this._playerObjects) {
                    this._playerObjects.forEach((player, name) => {
                      if (player.isAlive) {
                        // Enhanced floating animation for alive players
                        const floatOffset = Math.sin(time * 0.001 + player.baseAngle) * 0.2;
                        const bobOffset = Math.cos(time * 0.0015 + player.baseAngle * 2) * 0.05;
                        player.container.position.y = floatOffset + bobOffset;
                        
                        // More dynamic orb rotation
                        player.orb.rotation.y = time * 0.003;
                        player.orb.rotation.x = Math.sin(time * 0.002) * 0.15;
                        player.orb.rotation.z = Math.cos(time * 0.0025) * 0.1;
                        
                        // Enhanced glow animation
                        if (player.glow && player.glow.visible) {
                          player.glow.rotation.y = -time * 0.002;
                          const glowScale = 1 + Math.sin(time * 0.004 + player.baseAngle) * 0.15;
                          player.glow.scale.setScalar(glowScale);
                          
                          // Pulsing emissive intensity
                          const pulseIntensity = 0.3 + Math.sin(time * 0.005 + player.baseAngle) * 0.1;
                          player.glow.material.emissiveIntensity = pulseIntensity;
                        }
                        
                        // Enhanced pulse effect for active players
                        if (player.container.scale.x > 1.0) {
                          const pulseScale = 1.05 + Math.sin(time * 0.008) * 0.08;
                          player.container.scale.setScalar(pulseScale);
                        }
                        
                        // Enhanced breathing effect
                        if (player.body) {
                          const breathScale = 1 + Math.sin(time * 0.002 + player.baseAngle) * 0.03;
                          player.body.scale.y = breathScale;
                          if (player.shoulders) {
                            player.shoulders.scale.y = 0.6 * breathScale;
                          }
                        }
                        
                        // Subtle head movement
                        if (player.head) {
                          player.head.rotation.y = Math.sin(time * 0.001 + player.baseAngle) * 0.1;
                        }
                      } else {
                        // Dead players have reduced animation
                        if (player.orb) {
                          player.orb.rotation.y = time * 0.0008;
                        }
                      }
                    });
                  }

                  // Animate voting trails
                  if (this._animatingTrails) {
                      this._animatingTrails.forEach(trail => trail.update());
                  }

                  // Animate target rings
                  if (this._activeTargetRings) {
                      this._activeTargetRings.forEach((ringData, targetName) => {
                          const diff = ringData.targetOpacity - ringData.material.opacity;
                          if (Math.abs(diff) > 0.01) {
                              ringData.material.opacity += diff * 0.1;
                          } else if (ringData.targetOpacity === 0 && ringData.material.opacity > 0) {
                              this._targetRingsGroup.remove(ringData.ring);
                              this._activeTargetRings.delete(targetName);
                          }
                      });
                  }

                  // Use post-processing composer if available, otherwise fallback to direct render
                  if (this._composer) {
                    this._composer.render();
                  } else {
                    this._threejs.render(this._scene, this._camera);
                  }
                  this._labelRenderer.render(this._scene, this._camera);
                  this._RAF();
                });
              }
            }

            setupScene(BasicWorldDemo);
        } catch (error) {
            console.error("Failed to load Three.js modules:", error);
            parent.textContent = "Error loading 3D assets. Please refresh.";
        }
    };

    loadAndSetup();
  }

  function setupScene(BasicWorldDemo) {
    if (threeState.initialized) return;
    threeState.demo = new BasicWorldDemo({ parent, width, height });
    threeState.initialized = true;
  }

    function updateSceneFromGameState(gameState, playerMap, actingPlayerName) {
    if (!threeState.demo || !threeState.demo._playerObjects) return;

    const logUpToCurrentStep = gameState.eventLog;
    const lastEvent = logUpToCurrentStep.length > 0 ? logUpToCurrentStep[logUpToCurrentStep.length - 1] : null;

    // Determine correct phase from the last event log entry
    let phase = gameState.game_state_phase; // Default
    if (lastEvent && lastEvent.phase) {
        phase = lastEvent.phase;
    }

    // Update player statuses
    gameState.players.forEach(player => {
      const playerObj = threeState.demo._playerObjects.get(player.name);
      if (!playerObj) return;

      const threatLevel = gameState.playerThreatLevels.get(player.name) || 0;

      let primaryStatus = 'default'; // Default for alive players in daytime.
      if (!player.is_alive) {
        primaryStatus = 'dead';
      } else if (player.role === 'Werewolf' && phase.toUpperCase() === 'NIGHT') {
        primaryStatus = 'werewolf';
      } else if (player.role === 'Doctor' && phase.toUpperCase() === 'NIGHT') {
        primaryStatus = 'doctor';
      } else if (player.role === 'Seer' && phase.toUpperCase() === 'NIGHT') {
        primaryStatus = 'seer';
      }

      threeState.demo.updatePlayerStatus(player.name, primaryStatus, threatLevel);
    });

    // Update phase lighting
    threeState.demo.updatePhase(phase);

    // --- Vote Visualization Logic ---
    const currentVotes = new Map();

    // Find the start of the current voting/action session
    const lastNightStart = logUpToCurrentStep.findLastIndex(e => e.type === 'phase_divider' && e.divider === 'NIGHT START');
    const lastDayVoteStart = logUpToCurrentStep.findLastIndex(e => e.type === 'phase_divider' && e.divider === 'DAY VOTE START');
    const sessionStartIndex = Math.max(lastNightStart, lastDayVoteStart);

    let isVotingSession = false;
    if (sessionStartIndex > -1) {
        const lastOutcomeEventIndex = logUpToCurrentStep.findLastIndex(e => e.type === 'exile' || e.type === 'elimination' || e.type === 'save');
        // A session is active if it started after the last outcome, OR if the outcome is the current event.
        if (sessionStartIndex > lastOutcomeEventIndex || (lastOutcomeEventIndex > -1 && lastOutcomeEventIndex === logUpToCurrentStep.length - 1)) {
            isVotingSession = true;
        }
    }

    if (isVotingSession) {
        const alivePlayerNames = new Set(gameState.players.filter(p => p.is_alive).map(p => p.name));
        const relevantEvents = logUpToCurrentStep.slice(sessionStartIndex);
        for (const event of relevantEvents) {
            if (event.type === 'vote' || event.type === 'night_vote' || event.type === 'doctor_heal_action' || event.type === 'seer_inspection') {
                if (alivePlayerNames.has(event.actor_id)) {
                    currentVotes.set(event.actor_id, { target: event.target, type: event.type });
                }
            } else if (event.type === 'timeout') {
                currentVotes.delete(event.actor_id);
            }
        }
    }
    
    const clearVotingVisuals = !isVotingSession;
    threeState.demo.updateVoteVisuals(currentVotes, clearVotingVisuals);


    // Spotlight logic for night actions
    if (threeState.demo._spotLight) {
        const lastEvent = gameState.eventLog[gameState.eventLog.length - 1];
        const nightActor = (gameState.game_state_phase === 'NIGHT' && lastEvent && lastEvent.actor_id && ['WerewolfNightVoteDataEntry', 'DoctorHealActionDataEntry', 'SeerInspectActionDataEntry'].includes(lastEvent.dataType)) ? lastEvent.actor_id : null;

        if (nightActor) {
            const actorPlayer = threeState.demo._playerObjects.get(nightActor);
            if (actorPlayer) {
                const targetPosition = actorPlayer.container.position.clone();
                threeState.demo._spotLight.target.position.copy(targetPosition);
                threeState.demo._spotLight.position.set(targetPosition.x, targetPosition.y + 20, targetPosition.z + 5);
                threeState.demo._spotLight.visible = true;
            } else {
                threeState.demo._spotLight.visible = false;
            }
        } else {
            threeState.demo._spotLight.visible = false;
        }
    }

    // Handle animation for the current event actor
    if (lastEvent) {
        if (lastEvent.event_name === 'moderator_announcement') {
            // Moderator is speaking, expand all alive players
            gameState.players.forEach(player => {
                if (player.is_alive) {
                    threeState.demo.updatePlayerActive(player.name);
                }
            });
        } else if (lastEvent.actor_id && playerMap.has(lastEvent.actor_id)) {
            // A player is the actor
            const actorName = lastEvent.actor_id;
            threeState.demo.updatePlayerActive(actorName);

            // If the action was speaking, trigger the sound wave animation
            if (lastEvent.type === 'chat' && threeState.demo.triggerSpeakingAnimation) {
                threeState.demo.triggerSpeakingAnimation(actorName);
            }
        }
    }
  }

  // --- CSS for the UI ---
  const css = `
        /* Game Status Scoreboard */
        .game-scoreboard {
            position: fixed;
            top: 70px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999;
            background: linear-gradient(135deg, rgba(33, 40, 54, 0.95), rgba(44, 52, 68, 0.95));
            backdrop-filter: blur(15px);
            border: 1px solid rgba(116, 185, 255, 0.3);
            border-radius: 12px;
            padding: 12px 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            display: flex;
            gap: 20px;
            align-items: center;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            pointer-events: none;
        }
        
        .scoreboard-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 0 10px;
            border-right: 1px solid rgba(116, 185, 255, 0.2);
        }
        
        .scoreboard-item:last-child {
            border-right: none;
        }
        
        .scoreboard-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
            font-weight: 500;
        }
        
        .scoreboard-value {
            font-size: 1.1rem;
            color: var(--text-primary);
            font-weight: 600;
        }
        
        .scoreboard-value.alive {
            color: #00b894;
        }
        
        .scoreboard-value.dead {
            color: #e17055;
        }
        
        .scoreboard-value.werewolf {
            color: #e17055;
        }
        
        .scoreboard-value.villager {
            color: #74b9ff;
        }
        
        .scoreboard-action {
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.2), rgba(116, 185, 255, 0.1));
            border: 1px solid rgba(116, 185, 255, 0.3);
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 0.9rem;
            color: #74b9ff;
            font-weight: 500;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Phase Indicator */
        .phase-indicator {
            position: fixed;
            top: 60px;
            left: 50px;
            transform: translateX(-50%);
            z-index: 1000;
            padding: 12px 24px;
            border-radius: 30px;
            font-size: 1.2rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            pointer-events: none;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            scale: 0.6;
        }
        
        .phase-indicator.day {
            background: linear-gradient(135deg, rgba(255, 220, 100, 0.9), rgba(255, 180, 50, 0.9));
            color: #2d3436;
            border: 2px solid rgba(255, 255, 255, 0.5);
        }
        
        .phase-indicator.night {
            background: linear-gradient(135deg, rgba(30, 30, 60, 0.9), rgba(60, 60, 120, 0.9));
            color: #f8f9fa;
            border: 2px solid rgba(100, 100, 200, 0.5);
        }
        
        .phase-indicator .phase-icon {
            display: inline-block;
            margin-right: 8px;
            font-size: 1.4rem;
            vertical-align: middle;
        }
        
        :root {
            --night-bg: linear-gradient(135deg, #1a1a2e, #16213e);
            --day-bg: linear-gradient(135deg, #74b9ff, #0984e3);
            --night-text: #f8f9fa;
            --day-text: #2d3436;
            --dead-filter: grayscale(100%) brightness(40%) contrast(0.8);
            --active-border: #fdcb6e;
            --active-glow: rgba(253, 203, 110, 0.4);
            --werewolf-color: #e17055;
            --villager-color: #00b894;
            --doctor-color: #6c5ce7;
            --seer-color: #fd79a8;
            --panel-bg: rgba(33, 40, 54, 0.95);
            --panel-border: rgba(116, 185, 255, 0.2);
            --hover-bg: rgba(116, 185, 255, 0.1);
            --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            --text-primary: #f8f9fa;
            --text-secondary: #b2bec3;
            --text-muted: #74b9ff;
        }

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        .werewolf-parent {
            position: relative;
            overflow: hidden;
            width: 100%;
            height: 100%;
            background: radial-gradient(ellipse at center, #0f1419 0%, #000000 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .main-container {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 2;
            pointer-events: none;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--text-primary);
            font-weight: 400;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* Enhanced Panel Styling */
        .left-panel, .right-panel {
            position: fixed;
            top: 54px;
            max-height: calc(100vh - 124px);
            background: var(--panel-bg);
            backdrop-filter: blur(20px) saturate(1.5);
            border-radius: 16px;
            border: 1px solid var(--panel-border);
            padding: 20px;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            pointer-events: auto;
            box-shadow: var(--card-shadow), 0 0 40px rgba(116, 185, 255, 0.05);
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .left-panel {
            left: 20px;
            width: 320px;
        }
        
        .right-panel {
            right: 20px;
            width: 420px;
        }
        
        .left-panel:hover, .right-panel:hover {
            border-color: rgba(116, 185, 255, 0.3);
            box-shadow: var(--card-shadow), 0 0 60px rgba(116, 185, 255, 0.08);
        }
        
        /* Enhanced Headers */
        .right-panel h1, #player-list-area h1 {
            margin: 0 0 20px 0;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text-primary);
            position: relative;
            padding-bottom: 15px;
            flex-shrink: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        .right-panel h1 > span, #player-list-area h1 > span {
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .right-panel h1::after, #player-list-area h1::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, transparent, #74b9ff, transparent);
            border-radius: 2px;
        }

        #global-reasoning-toggle {
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            transition: all 0.2s ease;
        }
        #global-reasoning-toggle:hover {
            background-color: var(--hover-bg);
            color: var(--text-primary);
        }
        #global-reasoning-toggle svg {
            stroke: currentColor;
            width: 20px;
            height: 20px;
        }

        #global-audio-toggle {
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            transition: all 0.2s ease;
            font-size: 18px; /* For the emoji */
            vertical-align: middle; /* Align with the SVG icon */
            margin-left: 4px; /* Space from eye icon */
        }
        #global-audio-toggle:hover:not(.disabled) {
            background-color: var(--hover-bg);
            color: var(--text-primary);
        }
        #global-audio-toggle.disabled {
            color: #555; /* More dimmed */
            cursor: not-allowed;
            opacity: 0.5;
        }
        #global-audio-toggle.enabled {
            color: var(--text-primary); /* Brighter when enabled */
        }

        .reset-view-btn {
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            transition: all 0.2s ease;
            margin-left: 8px; /* Add some space */
        }
        .reset-view-btn:hover {
            background-color: var(--hover-bg);
            color: var(--text-primary);
        }
        .reset-view-btn svg {
            stroke: currentColor;
            width: 20px;
            height: 20px;
        }
        
        /* Enhanced Player List */
        #player-list-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        
        #player-list-container {
            overflow-y: auto;
            flex-grow: 1;
            padding-right: 8px;
            margin-right: -8px;
        }
        
        #player-list-container::-webkit-scrollbar {
            width: 6px;
        }
        
        #player-list-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }
        
        #player-list-container::-webkit-scrollbar-thumb {
            background: rgba(116, 185, 255, 0.3);
            border-radius: 3px;
        }
        
        #player-list-container::-webkit-scrollbar-thumb:hover {
            background: rgba(116, 185, 255, 0.5);
        }
        
        #player-list {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        /* Enhanced Player Cards */
        .player-card {
            position: relative;
            display: flex;
            align-items: center;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            overflow: hidden;
        }
        
        .player-card::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: transparent;
            transition: all 0.3s ease;
        }
        
        .player-card:hover {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.06));
            border-color: rgba(116, 185, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        }
        
        .player-card.active {
            background: linear-gradient(135deg, rgba(253, 203, 110, 0.15), rgba(253, 203, 110, 0.05));
            border-color: var(--active-border);
            box-shadow: 0 0 20px var(--active-glow), 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        .player-card.active::before {
            background: linear-gradient(180deg, var(--active-border), rgba(253, 203, 110, 0.5));
        }
        
        .player-card.dead {
            opacity: 0.5;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01));
            filter: brightness(0.7);
        }
        
        /* Enhanced Avatar */
        .avatar-container {
            position: relative;
            width: 40px;
            height: 40px;
            margin-right: 16px;
            flex-shrink: 0;
        }
        
        .player-card .avatar {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            object-fit: cover;
            background-color: #ffffff;
            border: 2px solid rgba(116, 185, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .player-card:hover .avatar {
            border-color: rgba(116, 185, 255, 0.4);
            box-shadow: 0 0 15px rgba(116, 185, 255, 0.2);
        }
        
        .player-card.active .avatar {
            border-color: var(--active-border);
            box-shadow: 0 0 15px var(--active-glow);
        }
        
        .player-card.dead .avatar {
            filter: var(--dead-filter);
            border-color: rgba(255, 255, 255, 0.1);
        }
        
        /* Enhanced Player Info */
        .player-info {
            flex-grow: 1;
            overflow: hidden;
            min-width: 0;
        }
        
        .player-name {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 4px;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            letter-spacing: -0.01em;
        }
        
        .player-role {
            font-size: 0.875rem;
            color: var(--text-secondary);
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .player-role.werewolf { color: var(--werewolf-color); }
        .player-role.villager { color: var(--villager-color); }
        .player-role.doctor { color: var(--doctor-color); }
        .player-role.seer { color: var(--seer-color); }

        .display-name {
            font-size: 0.8em;
            color: #888;
            margin-left: 5px;
        }
        
        /* Enhanced Threat Indicator */
        .threat-indicator {
            position: absolute;
            top: 12px;
            right: 12px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: transparent;
            transition: all 0.3s ease;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);
        }
        
        /* Enhanced Chat/Event Log */
        #chat-log {
            list-style: none;
            padding: 0;
            margin: 0;
            flex-grow: 1;
            overflow-y: auto;
            padding-right: 8px;
            margin-right: -8px;
        }
        
        #chat-log::-webkit-scrollbar {
            width: 6px;
        }
        
        #chat-log::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }
        
        #chat-log::-webkit-scrollbar-thumb {
            background: rgba(116, 185, 255, 0.3);
            border-radius: 3px;
        }
        
        #chat-log::-webkit-scrollbar-thumb:hover {
            background: rgba(116, 185, 255, 0.5);
        }

        #chat-log li.now-playing > .message-content > .balloon,
        #chat-log li.now-playing > .moderator-announcement-content,
        #chat-log li.now-playing.msg-entry {
            background: linear-gradient(135deg, rgba(253, 203, 110, 0.2), rgba(253, 203, 110, 0.1));
            border-color: #fdcb6e; /* A bright yellow */
            box-shadow: 0 0 5px rgba(253, 203, 110, 0.3);
            transition: all 0.2s ease-in-out;
        }

        /* Enhanced Chat Entries */
        .chat-entry {
            display: flex;
            margin-bottom: 20px;
            align-items: flex-start;
            animation: fadeInUp 0.3s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .chat-avatar {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            margin-right: 12px;
            object-fit: cover;
            flex-shrink: 0;
            border: 2px solid rgba(116, 185, 255, 0.2);
            transition: all 0.3s ease;
            background-color: #ffffff;
        }
        
        .chat-entry:hover .chat-avatar {
            border-color: rgba(116, 185, 255, 0.4);
        }
        
        .message-content {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
            min-width: 0;
        }
        
        /* Enhanced Message Bubbles */
        .balloon {
            padding: 14px 16px;
            border-radius: 16px 16px 16px 4px;
            max-width: 85%;
            word-wrap: break-word;
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.1), rgba(116, 185, 255, 0.05));
            border: 1px solid rgba(116, 185, 255, 0.2);
            transition: all 0.3s ease;
            position: relative;
            line-height: 1.4;
            font-size: 0.95rem;
        }
        
        .balloon:hover {
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.2), rgba(116, 185, 255, 0.1));
            border-color: rgba(116, 185, 255, 0.4);
            // transform: scale(1.01);
            transform: translateX(2px);
            cursor: pointer;
        }
        
        .chat-entry.event-day .balloon {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 193, 7, 0.05));
            border-color: rgba(255, 193, 7, 0.2);
            color: var(--text-primary);
        }
        
        .chat-entry.event-day .balloon:hover {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.2), rgba(255, 193, 7, 0.1));
            border-color: rgba(255, 193, 7, 0.3);
        }
        
        .chat-entry.event-night .balloon {
            background: linear-gradient(135deg, rgba(108, 92, 231, 0.1), rgba(108, 92, 231, 0.05));
            border-color: rgba(108, 92, 231, 0.2);
        }
        
        .event-log-list li.now-playing .balloon {
            background-color: #fcf8e3; /* A light yellow */
            border-color: #f7d794;
            transition: background-color 0.3s ease;
        }
        
        /* Enhanced System Messages */
        .msg-entry {
            border-left: 4px solid #f39c12;
            padding: 16px;
            margin: 16px 0;
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(243, 156, 18, 0.1), rgba(243, 156, 18, 0.05));
            border: 1px solid rgba(243, 156, 18, 0.2);
            transition: all 0.3s ease;
            animation: fadeInUp 0.3s ease-out;
        }
        
        .msg-entry:hover {
            background: linear-gradient(135deg, rgba(243, 156, 18, 0.15), rgba(243, 156, 18, 0.08));
            border-color: rgba(243, 156, 18, 0.3);
        }
        
        .msg-entry.event-day {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 193, 7, 0.05));
            border-color: rgba(255, 193, 7, 0.2);
        }
        
        .msg-entry.event-night {
            background: linear-gradient(135deg, rgba(108, 92, 231, 0.1), rgba(108, 92, 231, 0.05));
            border-color: rgba(108, 92, 231, 0.2);
        }
        
        .msg-entry.game-event {
            border-left-color: #e74c3c;
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(231, 76, 60, 0.05));
            border-color: rgba(231, 76, 60, 0.2);
        }
        
        .msg-entry.game-win {
            border-left-color: #2ecc71;
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(46, 204, 113, 0.05));
            border-color: rgba(46, 204, 113, 0.2);
            line-height: 1.6;
        }
        
        /* Enhanced Reasoning Text */
        .reasoning-text {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-style: italic;
            margin-top: 8px;
            padding-left: 12px;
            border-left: 2px solid rgba(116, 185, 255, 0.3);
            line-height: 1.4;
            font-family: 'JetBrains Mono', monospace;
            display: none;
        }
        .reasoning-text.visible {
            display: block;
        }
        .reasoning-toggle {
            cursor: pointer;
            font-size: 1rem;
            margin-left: 5;
            opacity: 0.6;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
        }
        .reasoning-toggle:hover {
            opacity: 1;
        }
        .msg-entry .reasoning-toggle {
            float: right;
        }
        
        /* Enhanced Citations */
        #chat-log cite {
            font-style: normal;
            font-weight: 600;
            display: flex;
            align-items: center;
            font-size: 0.9rem;
            color: var(--text-primary);
            margin-bottom: 6px;
            gap: 8px;
        }

        .cite-text-wrapper {
            display: flex;
            flex-direction: column;
        }
        
        /* Enhanced Moderator Announcements */
        .moderator-announcement {
            margin: 16px 0;
            animation: fadeInUp 0.3s ease-out;
        }
        
        .moderator-announcement-content {
            padding: 16px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(46, 204, 113, 0.05));
            border: 1px solid rgba(46, 204, 113, 0.2);
            border-left: 4px solid #2ecc71;
            color: var(--text-primary);
            line-height: 1.5;
            transition: all 0.3s ease;
        }
        
        .moderator-announcement-content:hover {
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(46, 204, 113, 0.08));
            border-color: rgba(46, 204, 113, 0.3);
        }
        
        /* Enhanced Timestamps */
        .timestamp {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(116, 185, 255, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: auto;
        }
        
        /* Enhanced Player Capsules */
        .player-capsule {
            display: inline-flex;
            align-items: center;
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.15), rgba(116, 185, 255, 0.08));
            border: 1px solid rgba(116, 185, 255, 0.2);
            border-radius: 16px;
            padding: 2px 10px 2px 2px;
            font-size: 0.875rem;
            font-weight: 500;
            margin: 0 2px;
            vertical-align: middle;
            transition: all 0.3s ease;
        }
        
        .player-capsule:hover {
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.2), rgba(116, 185, 255, 0.1));
            border-color: rgba(116, 185, 255, 0.3);
        }
        
        .capsule-avatar {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 6px;
            object-fit: cover;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background-color: #ffffff;
        }

        .capsule-display-name {
            font-size: 0.9em;
            color: #888;
            margin-left: 5px;
        }
        
        /* Enhanced TTS Button */
        .tts-button {
            cursor: pointer;
            font-size: 1.1rem;
            margin-left: 12px;
            padding: 4px;
            border-radius: 50%;
            transition: all 0.3s ease;
            opacity: 0.6;
        }
        
        .tts-button:hover {
            opacity: 1;
            background: rgba(116, 185, 255, 0.1);
            transform: scale(1.1);
        }
        
        /* Enhanced Audio Controls */
        .audio-controls {
            padding: 16px 0;
            border-top: 1px solid rgba(116, 185, 255, 0.2);
            margin-top: 16px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            padding: 16px;
        }
        
        .audio-controls label {
            display: block;
            margin-bottom: 8px;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
        }
        
        .audio-controls input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(116, 185, 255, 0.2);
            outline: none;
            -webkit-appearance: none;
        }
        
        .audio-controls input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #74b9ff;
            cursor: pointer;
            border: 2px solid #ffffff;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }
        
        #pause-audio {
            background-color: rgba(116, 185, 255, 0.1);
            border: 1px solid rgba(116, 185, 255, 0.3);
            border-radius: 50%;
            width: 36px;
            height: 36px;
            cursor: pointer;
            padding: 0;
            background-size: 16px;
            background-repeat: no-repeat;
            background-position: center;
            transition: all 0.3s ease;
            filter: none;
        }
        
        #pause-audio:hover {
            background-color: rgba(116, 185, 255, 0.2);
            border-color: rgba(116, 185, 255, 0.5);
            transform: scale(1.1);
        }
        
        #pause-audio.paused {
            background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM3NGI5ZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWdvbiBwb2ludHM9IjYgMyAyMCAxMiA2IDIxIi8+PC9zdmc+');
        }
        
        #pause-audio.playing {
            background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM3NGI5ZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cmVjdCB4PSI2IiB5PSI0IiB3aWR0aD0iNCIgaGVpZ2h0PSIxNiIgcng9IjEiLz48cmVjdCB4PSIxNCIgeT0iNCIgd2lkdGg9IjQiIGhlaWdodD0iMTYiIHJ4PSIxIi8+PC9zdmc+');
        }
        
        /* Message text formatting */
        .msg-text {
            line-height: 1.5;
            font-size: 0.95rem;
        }
        
        .msg-text br {
            display: block;
            margin-bottom: 0.5em;
            content: "";
        }
        
        /* Smooth scrolling */
        * {
            scrollbar-width: thin;
            scrollbar-color: rgba(116, 185, 255, 0.3) transparent;
        }
    `;

  // --- TTS Management ---
  const audioMap = window.AUDIO_MAP || {};

  if (!window.kaggleWerewolf) {
      window.kaggleWerewolf = {
          audioQueue: [],
          isAudioPlaying: false,
          isAudioEnabled: false,
          isPaused: false,
          lastPlayedStep: parseInt(sessionStorage.getItem('ww_lastPlayedStep') || '-1', 10),
          audioPlayer: new Audio(),
          playbackRate: 1.6,
          allEvents: null,
          audioContextActivated: false,
      };
  }
  const audioState = window.kaggleWerewolf;

  if (audioState.hasAudioTracks === undefined) { 
      audioState.hasAudioTracks = Object.keys(audioMap).length > 0; 
  }

  function setPlaybackRate(rate) {
      audioState.playbackRate = rate;
      if (audioState.isAudioPlaying) {
          audioState.audioPlayer.playbackRate = rate;
      }
  }

  function speak(allEventsIndex) {
    if (allEventsIndex === undefined) return;

    // 1. Find the corresponding display step for the slider.
    const displayStep = window.werewolfGamePlayer.allEventsIndexToDisplayStep[allEventsIndex];

    // 2. Jump the slider.
    // This will automatically trigger our setStep wrapper, which calls
    // stopAndClearAudio() and sets audioState.isPaused = true.
    if (displayStep !== undefined && window.kaggle && window.kaggle.setStep) {
        // window.kaggle.setStep(displayStep);
        context.__mainContext.setStep(displayStep);
    }

    // 3. Start continuous playback from that point.
    // playAudioFrom() will see we are paused, load the new queue,
    // and immediately start playing.
    playAudioFrom(allEventsIndex);
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

  /**
  * Creates a memoized function to replace player IDs with HTML capsules.
  * This function pre-computes and caches sorted player data for efficiency.
  * @param {Map<string, object>} playerMap - A map from player ID to player object.
  * @returns {function(string): string} A function that takes text and returns it with player IDs replaced.
  */
  function createPlayerIdReplacer(playerMap) {
    // Cache for already processed text strings (memoization)
    const textCache = new Map();

    // --- Pre-computation Cache ---
    const sortedPlayerReplacements = [...playerMap.keys()]
        .sort((a, b) => b.length - a.length) // Sort by length to match longest names first
        .map(playerId => {
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
                regex: new RegExp(`(^|[^\\w.-])(${playerId.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')})(\\.?)(?![\\w-])`, 'g')
            };
        }).filter(Boolean);

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

  function createPlayerCapsule(player) {
    if (!player) return '';
    let display_name_elem = (player.display_name && (player.name !== player.display_name)) ? `<span class="capsule-display-name">${player.display_name}</span>` : "";
    return `<span class="player-capsule" title="${player.name}">
        <img src="${player.thumbnail}" class="capsule-avatar" alt="${player.name}">
        <span class="capsule-name">${player.name}</span>${display_name_elem}
    </span>`;
  }

  function replacePlayerIdsWithCapsules(text, playerIds, playerMap) {
    if (!text) return '';
    if (!playerIds || playerIds.length === 0) {
        return text;
    }
    let newText = text;
    const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

    sortedPlayerIds.forEach(playerId => {
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

  function replacePlayerIdsWithBold(text, playerIds) {
    if (!text) return '';
    if (!playerIds || playerIds.length === 0) {
        return text;
    }
    let newText = text;
    const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

    sortedPlayerIds.forEach(playerId => {
        const regex = new RegExp(`\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\b`, 'g');
        newText = newText.replace(regex, `<strong>${playerId}</strong>`);
    });
    return newText;
  }


  function getThreatColor(threatLevel) {
    const value = Math.max(0, Math.min(1, threatLevel));
    const hue = 120 * (1 - value);
    return `hsl(${hue}, 100%, 50%)`;
  }

  function updatePlayerList(container, gameState, actingPlayerName) {
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
        let player_name_element = `<div class="player-name" title="${player.name}">${player.name}</div>`
        if (player.display_name && player.display_name !== player.name) {
            player_name_element = `<div class="player-name" title="${player.name}">
                ${player.name}<span class="display-name">${player.display_name}</span>
            </div>`
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

  function updateEventLog(container, gameState, playerMap) {
    const audioState = window.kaggleWerewolf;
    const audioToggleDisabled = !audioState.hasAudioTracks;
    const audioToggleEnabled = audioState.isAudioEnabled && !audioToggleDisabled;
    const audioToggleTitle = audioToggleDisabled ? 'Audio Not Available' : 'Toggle Audio';
    const audioToggleIcon = audioToggleEnabled ? '&#x1F50A;' : '&#x1F507;'; // Speaker vs Muted
    const audioToggleClasses = `audio-toggle-btn ${audioToggleDisabled ? 'disabled' : ''} ${audioToggleEnabled ? 'enabled' : ''}`;

    container.innerHTML = `
        <h1>
            <span>Events</span>
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

    const logUl = document.createElement('ul');
    logUl.id = 'chat-log';

    const logEntries = gameState.eventLog;

    if (logEntries.length === 0) {
        const li = document.createElement('li');
        li.className = 'msg-entry';
        li.innerHTML = `<cite>System</cite><div>The game is about to begin...</div>`;
        logUl.appendChild(li);
    } else {
        logEntries.forEach( (entry, entryIndex) => {
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
                        const cleanedContent = listContent.replace(/'/g, "").replace(/, /g, " ").trim();
                        
                        // If punctuation was captured, return the content with a space before the punctuation
                        if (punctuation) {
                            return cleanedContent + " " + punctuation.trim();
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
                    let role_text = (entry.role) ? ` (${entry.role})` : "";
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
                    let elim_role_text = (entry.role) ? ` Their role was a ${entry.role}.` : "";
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
        globalToggle.addEventListener('click', (event) => {
            event.stopPropagation();
            const reasoningTexts = logUl.querySelectorAll('.reasoning-text');
            if (reasoningTexts.length === 0) return;

            // Determine if we should show or hide all. If any are visible, we hide all. Otherwise, show all.
            const shouldShow = ![...reasoningTexts].some(el => el.classList.contains('visible'));

            reasoningTexts.forEach(el => {
                el.classList.toggle('visible', shouldShow);
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
                    const audio = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
                    audio.play().catch(e => console.warn("Audio context activation failed:", e));
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
            if(speedLabel) speedLabel.textContent = newRate.toFixed(1) + 'x';
        });
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

    // --- Main Rendering Logic (Incremental) ---
    // Only create UI elements if they don't exist
    let mainContainer = parent.querySelector('.main-container');
    let style = parent.querySelector('style');
    
    if (!style) {
        style = document.createElement('style');
        style.textContent = css;
        parent.appendChild(style);
    }

    initThreeJs();

    if (!environment || !environment.steps || environment.steps.length === 0 || step >= environment.steps.length) {
        if (!mainContainer) {
            const tempContainer = document.createElement("div");
            tempContainer.textContent = "Waiting for game data or invalid step...";
            parent.appendChild(tempContainer);
        }
        return;
    }

    // Initialize player mapping for 3D scene
    let playerNamesFor3D = [];
    let playerThumbnailsFor3D = {};

    // --- State Reconstruction ---
    const player = window.werewolfGamePlayer;
    const { allEvents, displayStepToAllEventsIndex, originalSteps, eventToKaggleStep } = player;

    if (step >= displayStepToAllEventsIndex.length) {
      console.error("Step is out of bounds for displayStepToAllEventsIndex", step, displayStepToAllEventsIndex.length);
      return;
    }
    const allEventsIndex = displayStepToAllEventsIndex[step];
    const eventStep = allEventsIndex; // for clarity
    const kaggleStep = eventToKaggleStep[eventStep] || 0;

    let gameState = {
        players: [],
        day: 0,
        phase: 'GAME_SETUP',
        game_state_phase: 'DAY',
        gameWinner: null,
        eventLog: [],
        playerThreatLevels: new Map()
    };

    const firstObs = originalSteps[0]?.[0]?.observation?.raw_observation;
    let allPlayerNamesList;
    let playerThumbnails = {};

    if (firstObs && firstObs.all_player_ids) {
        allPlayerNamesList = firstObs.all_player_ids;
        playerThumbnails = firstObs.player_thumbnails || {};
        playerNamesFor3D = [...allPlayerNamesList];
        playerThumbnailsFor3D = {...playerThumbnails};
    } else if (environment.configuration && environment.configuration.agents) {
        // console.warn("Renderer: Initial observation missing or incomplete. Reconstructing players from configuration.");
        allPlayerNamesList = environment.configuration.agents.map(agent => agent.id);
        environment.configuration.agents.forEach(agent => {
            if (agent.id && agent.thumbnail) {
                playerThumbnails[agent.id] = agent.thumbnail;
            }
        });
        playerNamesFor3D = [...allPlayerNamesList];
        playerThumbnailsFor3D = {...playerThumbnails};
    }

    if (!allPlayerNamesList || allPlayerNamesList.length === 0) {
        const tempContainer = document.createElement("div");
        tempContainer.textContent = "Waiting for game data: No players found in observation or configuration.";
        parent.appendChild(tempContainer);
        return;
    }

    gameState.players = environment.configuration.agents.map( agent => ({
        name: agent.id, is_alive: true, role: agent.role, team: 'Unknown',
        status: 'Alive', thumbnail: agent.thumbnail || `https://via.placeholder.com/40/2c3e50/ecf0f1?text=${agent.id.charAt(0)}`,
        display_name: agent.display_name
    }));
    const playerMap = new Map(gameState.players.map(p => [p.name, p]));

    // Initialize and cache the replacer function if it doesn't exist
    if (!player.playerIdReplacer) {
        player.playerIdReplacer = createPlayerIdReplacer(playerMap);
    }

    gameState.players.forEach(p => gameState.playerThreatLevels.set(p.name, 0));

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

    function threatStringToLevel(threatString) {
        switch(threatString) {
            case 'SAFE': return 0;
            case 'UNEASY': return 0.5;
            case 'DANGER': return 1.0;
            default: return 0;
        }
    }

    // Reconstruct state up to current kaggleStep
    for (let s = 0; s <= kaggleStep; s++) {
        const stepStateList = originalSteps[s];
        if (!stepStateList) continue;

        const currentObsForStep = stepStateList[0]?.observation?.raw_observation;
        if (currentObsForStep) {
            gameState.day = currentObsForStep.day;
            gameState.phase = currentObsForStep.phase;
            gameState.game_state_phase = currentObsForStep.game_state_phase;
        }
    }

    // Populate event log up to current eventStep
    for (let i = 0; i <= eventStep; i++) {
        const historyEvent = allEvents[i];
        const data = historyEvent.data;
        const timestamp = historyEvent.created_at;

        if (data && data.actor_id && data.perceived_threat_level) {
            const threatScore = threatStringToLevel(data.perceived_threat_level);
            gameState.playerThreatLevels.set(data.actor_id, threatScore);
        }

        if (!data) {
            if (historyEvent.event_name === 'vote_action') {
                const match = historyEvent.description.match(/P(player_\d+)/);
                if (match) {
                    const actor_id = match[1];
                    gameState.eventLog.push({ type: 'timeout', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, actor_id: actor_id, reasoning: "Timed out", timestamp: historyEvent.created_at });
                }
            } else if (historyEvent.event_name === 'day_start' || historyEvent.event_name === 'night_start') {
                gameState.eventLog.push({ type: 'system', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, text: historyEvent.description, allEventsIndex: i, timestamp});
            }
            continue;
        }

         switch (historyEvent.dataType) {
            case 'ChatDataEntry':
                gameState.eventLog.push({ type: 'chat', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, speaker: data.actor_id, message: data.message, reasoning: data.reasoning, timestamp, allEventsIndex: i, mentioned_player_ids: data.mentioned_player_ids || [] });
                break;
            case 'DayExileVoteDataEntry':
                gameState.eventLog.push({ type: 'vote', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, allEventsIndex: i, timestamp });
                break;
            case 'WerewolfNightVoteDataEntry':
                gameState.eventLog.push({ type: 'night_vote', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, allEventsIndex: i, timestamp });
                break;
            case 'DoctorHealActionDataEntry':
                gameState.eventLog.push({ type: 'doctor_heal_action', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, allEventsIndex: i, timestamp });
                break;
            case 'SeerInspectActionDataEntry':
                gameState.eventLog.push({ type: 'seer_inspection', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, allEventsIndex: i, timestamp });
                break;
            case 'DayExileElectedDataEntry':
                gameState.eventLog.push({ type: 'exile', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, name: data.elected_player_id, role: data.elected_player_role_name, allEventsIndex: i, timestamp });
                break;
            case 'WerewolfNightEliminationDataEntry':
                gameState.eventLog.push({ type: 'elimination', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, name: data.eliminated_player_id, role: data.eliminated_player_role_name, allEventsIndex: i, timestamp });
                break;
            case 'SeerInspectResultDataEntry':
                gameState.eventLog.push({ type: 'seer_inspection_result', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, seer: data.actor_id, target: data.target_id, role: data.role, team: data.team, allEventsIndex: i, timestamp });
                break;
            case 'DoctorSaveDataEntry':
                gameState.eventLog.push({ type: 'save', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, saved_player: data.saved_player_id, allEventsIndex: i, timestamp });
                break;
            case 'PhaseDividerDataEntry':
                gameState.eventLog.push({ type: 'phase_divider', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, divider: data.divider_type, allEventsIndex: i, timestamp });
                break;
            case 'GameEndResultsDataEntry':
                gameState.gameWinner = data.winner_team;
                const winners = gameState.players.filter(p => p.team === data.winner_team).map(p => p.name);
                const losers = gameState.players.filter(p => p.team !== data.winner_team).map(p => p.name);
                gameState.eventLog.push({ type: 'game_over', step: historyEvent.kaggleStep, day: Infinity, phase: 'GAME_OVER', winner: data.winner_team, winners, losers, allEventsIndex: i, timestamp });
                break;
            case 'DiscussionOrderDataEntry':
                gameState.eventLog.push({ type: 'system', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, text: historyEvent.description, allEventsIndex: i, timestamp });
                break;
            default:
                if (systemEntryTypeSet.has(historyEvent.event_name)) {
                    gameState.eventLog.push({ type: 'system', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, text: historyEvent.description, allEventsIndex: i, timestamp, data: data});
                }
                break;
         }
    }

    if (eventStep < audioState.lastPlayedStep) {
        audioState.audioQueue = [];
        audioState.isAudioPlaying = false;
        if (audioState.audioPlayer) {
            audioState.audioPlayer.pause();
        }
        const chatLog = parent.querySelector('#chat-log');
        if (chatLog) {
            chatLog.innerHTML = '';
        }
    }

    audioState.lastPlayedStep = eventStep;
    sessionStorage.setItem('ww_lastPlayedStep', eventStep);

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

    // 1. Get the actual event being displayed at this step.
    const currentEvent = allEvents[eventStep]; 
    let nameToHighlight = null; // Initialize to null as requested.

    if (currentEvent) {
        // 2. Check for a player actor in the event's data (covers chat, votes, night actions).
        if (currentEvent.data && currentEvent.data.actor_id) {
            nameToHighlight = currentEvent.data.actor_id;
        } 
        // 3. Handle the special case for a timeout.
        else if (currentEvent.event_name === 'vote_action' && !currentEvent.data) {
            const match = currentEvent.description.match(/P(player_\d+)/);
            if (match && playerMap.has(match[1])) {
                nameToHighlight = match[1];
            }
        }
    }

    Object.assign(parent.style, { width: `${width}px`, height: `${height}px` });
    parent.className = 'werewolf-parent';

    // Create or get existing main container
    if (!mainContainer) {
        mainContainer = document.createElement('div');
        mainContainer.className = 'main-container';
        parent.appendChild(mainContainer);
    }
    
    // Create or update phase indicator
    let phaseIndicator = parent.querySelector('.phase-indicator');
    if (!phaseIndicator) {
        phaseIndicator = document.createElement('div');
        phaseIndicator.className = 'phase-indicator';
        parent.appendChild(phaseIndicator);
    }
    
    // Update phase indicator based on current game state
    const currentPhase = allEvents[eventStep].phase.toUpperCase() || 'DAY';
    const isNight = currentPhase === 'NIGHT';
    phaseIndicator.className = `phase-indicator ${isNight ? 'night' : 'day'}`;
    if (allEvents[eventStep]?.event_name == 'game_end') {
        phaseIndicator.innerHTML = `
        <span class="phase-icon">${isNight ? '&#x1F319;' : '&#x2600;'}</span>
        `;
    } else {
        phaseIndicator.innerHTML = `
        <span class="phase-icon">${isNight ? '&#x1F319;' : '&#x2600;'}</span>
        <span>${allEvents[eventStep].day}</span>
        `;
    }
    
    // Create or update game scoreboard
    let scoreboard = parent.querySelector('.game-scoreboard');
    if (!scoreboard) {
        scoreboard = document.createElement('div');
        scoreboard.className = 'game-scoreboard';
        parent.appendChild(scoreboard);
    }
    
    // Calculate game statistics
    const alivePlayers = gameState.players.filter(p => p.is_alive).length;
    const deadPlayers = gameState.players.filter(p => !p.is_alive).length;
    const werewolves = gameState.players.filter(p => p.is_alive && p.role === 'Werewolf').length;
    const villagers = gameState.players.filter(p => p.is_alive && p.role !== 'Werewolf' && p.role !== 'Unknown').length;
    
    // Determine current action based on phase and recent events
    let currentAction = 'Waiting...';
    const lastEvent = gameState.eventLog[gameState.eventLog.length - 1];
    
    if (gameState.gameWinner) {
        currentAction = `${gameState.gameWinner} Win!`;
    } else if (gameState.phase === 'VOTING') {
        currentAction = 'Voting Phase';
    } else if (gameState.phase === 'DISCUSSION') {
        currentAction = 'Discussion';
    } else if (isNight) {
        // Check for recent night actions
        if (lastEvent) {
            if (lastEvent.type === 'night_vote') {
                currentAction = 'Werewolves Voting';
            } else if (lastEvent.type === 'doctor_heal_action') {
                currentAction = 'Doctor Saving';
            } else if (lastEvent.type === 'seer_inspection') {
                currentAction = 'Seer Inspecting';
            } else {
                currentAction = 'Night Actions';
            }
        } else {
            currentAction = 'Night Phase';
        }
    } else {
        // Day phase
        if (lastEvent && lastEvent.type === 'chat') {
            currentAction = 'Discussion';
        } else if (lastEvent && lastEvent.type === 'vote') {
            currentAction = 'Exile Voting';
        } else {
            currentAction = 'Day Phase';
        }
    }
    
    // Update scoreboard content
    scoreboard.innerHTML = `
        <div class="scoreboard-item">
            <div class="scoreboard-label">Day</div>
            <div class="scoreboard-value">${gameState.day || 0}</div>
        </div>
        <div class="scoreboard-item">
            <div class="scoreboard-label">Alive</div>
            <div class="scoreboard-value alive">${alivePlayers}</div>
        </div>
        <div class="scoreboard-item">
            <div class="scoreboard-label">Out</div>
            <div class="scoreboard-value dead">${deadPlayers}</div>
        </div>
        ${werewolves > 0 || villagers > 0 ? `
            <div class="scoreboard-item">
                <div class="scoreboard-label">Werewolves</div>
                <div class="scoreboard-value werewolf">${werewolves}</div>
            </div>
            <div class="scoreboard-item">
                <div class="scoreboard-label">Villagers</div>
                <div class="scoreboard-value villager">${villagers}</div>
            </div>
        ` : ''}
        <div class="scoreboard-item">
            <div class="scoreboard-action">${currentAction}</div>
        </div>
    `;

    // Create or get existing panels
    let leftPanel = mainContainer.querySelector('.left-panel');
    if (!leftPanel) {
        leftPanel = document.createElement('div');
        leftPanel.className = 'left-panel';
        mainContainer.appendChild(leftPanel);
    }

    let playerListArea = leftPanel.querySelector('#player-list-area');
    if (!playerListArea) {
        playerListArea = document.createElement('div');
        playerListArea.id = 'player-list-area';
        leftPanel.appendChild(playerListArea);
    }

    let rightPanel = mainContainer.querySelector('.right-panel');
    if (!rightPanel) {
        rightPanel = document.createElement('div');
        rightPanel.className = 'right-panel';
        mainContainer.appendChild(rightPanel);
    }

    // Update existing content instead of clearing and rebuilding
    updatePlayerList(playerListArea, gameState, nameToHighlight);
    updateEventLog(rightPanel, gameState, playerMap);

    // Update 3D scene based on game state
    updateSceneFromGameState(gameState, playerMap, nameToHighlight);
    
    // Initialize 3D players if needed
    if (threeState.demo && threeState.demo._playerObjects && threeState.demo._playerObjects.size === 0 && playerNamesFor3D.length > 0) {
        initializePlayers3D(gameState, playerNamesFor3D, playerThumbnailsFor3D, threeState);
    }
}

function initializePlayers3D(gameState, playerNames, playerThumbnails, threeState) {
    if (!threeState || !threeState.demo || !threeState.demo._playerObjects) return;
    
    // Clear existing player objects
    if (threeState.demo._playerGroup) {
        // Remove all children from the group
        while(threeState.demo._playerGroup.children.length > 0) {
            threeState.demo._playerGroup.remove(threeState.demo._playerGroup.children[0]);
        }
    }
    threeState.demo._playerObjects.clear();
    
    const numPlayers = playerNames.length;
    const radius = 18; // Increased radius to use more space
    const playerHeight = 4;
    
    const THREE = threeState.demo._THREE;
    const CSS2DObject = threeState.demo._CSS2DObject;
    
    // Create a circular platform
    const platformGeometry = new THREE.RingGeometry(radius - 2, radius + 3, 64);
    const platformMaterial = new THREE.MeshStandardMaterial({
        color: 0x2a2a3a,
        roughness: 0.9,
        metalness: 0.1,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide
    });
    const platform = new THREE.Mesh(platformGeometry, platformMaterial);
    platform.rotation.x = -Math.PI / 2;
    platform.position.y = -0.05;
    platform.receiveShadow = true;
    threeState.demo._playerGroup.add(platform);
    
    // Create center decoration
    const centerGeometry = new THREE.CylinderGeometry(3, 3, 0.2, 32);
    const centerMaterial = new THREE.MeshStandardMaterial({
        color: 0x444466,
        roughness: 0.7,
        metalness: 0.3,
        emissive: 0x222244,
        emissiveIntensity: 0.2
    });
    const centerPlatform = new THREE.Mesh(centerGeometry, centerMaterial);
    centerPlatform.position.y = 0.1;
    centerPlatform.receiveShadow = true;
    threeState.demo._playerGroup.add(centerPlatform);
    
    // Add decorative lines from center to each player position
    const linesMaterial = new THREE.LineBasicMaterial({
        color: 0x334455,
        transparent: true,
        opacity: 0.3
    });
    
    playerNames.forEach((name, i) => {
        const displayName = gameState.players[i].display_name || '';
        const playerContainer = new THREE.Group();
        // Use full circle (360 degrees)
        const angle = (i / numPlayers) * Math.PI * 2;
        
        const x = radius * Math.sin(angle);
        const z = radius * Math.cos(angle);
        playerContainer.position.set(x, 0, z);
        
        // Create line from center to player
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0.05, 0),
            new THREE.Vector3(x, 0.05, z)
        ]);
        const line = new THREE.Line(lineGeometry, linesMaterial);
        threeState.demo._playerGroup.add(line);
        
        // Create pedestal for each player
        const pedestalGeometry = new THREE.CylinderGeometry(1.5, 1.8, 0.4, 16);
        const pedestalMaterial = new THREE.MeshStandardMaterial({
            color: 0x333344,
            roughness: 0.8,
            metalness: 0.2,
            emissive: 0x111122,
            emissiveIntensity: 0.1
        });
        const pedestal = new THREE.Mesh(pedestalGeometry, pedestalMaterial);
        pedestal.position.y = 0.2;
        pedestal.castShadow = true;
        pedestal.receiveShadow = true;
        playerContainer.add(pedestal);
        
        // Create player body (more detailed)
        const bodyGeometry = new THREE.CylinderGeometry(0.8, 1, playerHeight * 0.6, 16);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0x4466ff,
            roughness: 0.5,
            metalness: 0.3,
            emissive: 0x111166,
            emissiveIntensity: 0.2
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = playerHeight * 0.4;
        body.castShadow = true;
        body.receiveShadow = true;
        playerContainer.add(body);
        
        // Create shoulders
        const shoulderGeometry = new THREE.SphereGeometry(1, 16, 8);
        const shoulderMaterial = new THREE.MeshStandardMaterial({
            color: 0x4466ff,
            roughness: 0.5,
            metalness: 0.3,
            emissive: 0x111166,
            emissiveIntensity: 0.2
        });
        const shoulders = new THREE.Mesh(shoulderGeometry, shoulderMaterial);
        shoulders.position.y = playerHeight * 0.65;
        shoulders.scale.set(1.2, 0.6, 0.8);
        shoulders.castShadow = true;
        playerContainer.add(shoulders);
        
        // Create player head (sphere)
        const headGeometry = new THREE.SphereGeometry(0.7, 16, 16);
        const headMaterial = new THREE.MeshStandardMaterial({
            color: 0xfdbcb4,
            roughness: 0.7,
            metalness: 0.1,
            emissive: 0x442211,
            emissiveIntensity: 0.1
        });
        const head = new THREE.Mesh(headGeometry, headMaterial);
        head.position.y = playerHeight * 0.85;
        head.castShadow = true;
        head.receiveShadow = true;
        playerContainer.add(head);
        
        // Create eyes
        const eyeGeometry = new THREE.SphereGeometry(0.08, 8, 6);
        const eyeMaterial = new THREE.MeshStandardMaterial({
            color: 0x000000,
            roughness: 0.3,
            metalness: 0.8
        });
        const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        leftEye.position.set(-0.2, playerHeight * 0.87, 0.6);
        playerContainer.add(leftEye);
        
        const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        rightEye.position.set(0.2, playerHeight * 0.87, 0.6);
        playerContainer.add(rightEye);
        
        // Create glowing orb for status (more dramatic)
        const orbGeometry = new THREE.IcosahedronGeometry(0.3, 2);
        const orbMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ff00,
            emissive: 0x00ff00,
            emissiveIntensity: 0.8,
            transparent: true,
            opacity: 0.9
        });
        const orb = new THREE.Mesh(orbGeometry, orbMaterial);
        orb.position.y = playerHeight * 1.2;
        orb.name = 'statusOrb';
        playerContainer.add(orb);
        
        // Add outer glow sphere
        const glowGeometry = new THREE.SphereGeometry(0.5, 12, 8);
        const glowMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ff00,
            emissive: 0x00ff00,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.3
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.y = playerHeight * 1.2;
        playerContainer.add(glow);
        
        // Add point light for glow effect
        const orbLight = new THREE.PointLight(0x00ff00, 0.8, 8);
        orbLight.position.y = playerHeight * 1.2;
        orbLight.name = 'orbLight';
        orbLight.castShadow = true;
        playerContainer.add(orbLight);
        
        // Make player face center without flipping
        // Calculate the angle to face the center
        playerContainer.rotation.y = -angle + Math.PI / 2;
        
        // Create nameplate with actual player thumbnail
        const thumbnailUrl = playerThumbnails[name] || `https://via.placeholder.com/60/2c3e50/ecf0f1?text=${name.charAt(0)}`;
        const nameplate = threeState.demo._createNameplate(name, displayName, thumbnailUrl, CSS2DObject);
        nameplate.position.set(0, playerHeight * 2.0, 0);
        playerContainer.add(nameplate);
        
        // Store references
        threeState.demo._playerObjects.set(name, {
            container: playerContainer,
            body: body,
            head: head,
            shoulders: shoulders,
            orb: orb,
            glow: glow,
            orbLight: orbLight,
            nameplate: nameplate,
            pedestal: pedestal,
            originalPosition: playerContainer.position.clone(),
            baseAngle: angle,
            isAlive: true
        });
        
        threeState.demo._playerGroup.add(playerContainer);
    });
    
    // Adjust camera to see the full circle
    if (threeState.demo._camera) {
        threeState.demo._camera.position.set(25, 30, 25);
        threeState.demo._controls.target.set(0, 5, 0);
        threeState.demo._controls.update();
    }
}
