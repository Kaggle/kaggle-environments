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
      players3DInitialized: false,  // Add flag to track 3D player initialization
      deathAnimationCompleted: new Map()  // Add persistent Map to track death animations
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
            // Import THREE as a module
            const THREEModule = await import('https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js');
            const THREE = THREEModule.default || THREEModule;
            
            // Make THREE available globally for VolumetricFire
            window.THREE = THREE;
            
            const { OrbitControls } = await import('https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js');
            const { FBXLoader } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/FBXLoader.js');
            const { SkeletonUtils } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/utils/SkeletonUtils.js');
            const { EXRLoader } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/EXRLoader.js');
            const { CSS2DRenderer, CSS2DObject } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/renderers/CSS2DRenderer.js');
            const { EffectComposer } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/EffectComposer.js');
            const { RenderPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/RenderPass.js');
            const { UnrealBloomPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/UnrealBloomPass.js');
            const { ShaderPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/ShaderPass.js');
            const { FilmPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/FilmPass.js');
            const { Sky } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/objects/Sky.js');
            
            // Verify THREE is properly set on window before loading VolumetricFire
            if (!window.THREE || !window.THREE.WebGLRenderer) {
                throw new Error('THREE library not properly loaded on window object');
            }
            
            // Load VolumetricFire library
            const VolumetricFireModule = await import('/experiment/static/volumetric_fire/VolumetricFire.js');
            const VolumetricFire = VolumetricFireModule.default || window.VolumetricFire;

            class BasicWorldDemo {
              constructor(options) {
                this._Initialize(options, THREE, OrbitControls, FBXLoader, SkeletonUtils, CSS2DRenderer, CSS2DObject, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass, Sky, VolumetricFire, EXRLoader);
              }

              _Initialize(options, THREE, OrbitControls, FBXLoader, SkeletonUtils, CSS2DRenderer, CSS2DObject, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass, Sky, VolumetricFire) {
                this._parent = options.parent;
                this._width = options.width;
                this._height = options.height;

                // Initialize FBXLoader and SkeletonUtils
                this._fbxLoader = new FBXLoader();
                this._skeletonUtils = SkeletonUtils;

                this._EXRLoader = EXRLoader
                
                // Store VolumetricFire reference
                this._VolumetricFire = VolumetricFire;

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
                // this._threejs.toneMappingExposure = 1.05; // Set to visible default value
                this._threejs.toneMappingExposure = 1.0;
                
                console.error('[SKY DEBUG] Renderer settings:', {
                  toneMapping: 'ACESFilmicToneMapping',
                  toneMappingExposure: this._threejs.toneMappingExposure,
                  outputEncoding: 'sRGBEncoding'
                });
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
                
                // Add subtle atmospheric fog for depth
                // Using FogExp2 for exponential fog that's more visible at ground level
                // this._scene.fog = new THREE.FogExp2(0x87CEEB, 0.01); // Light blue-grey, very low density

                this._createAdvancedSkySystem(THREE, Sky);
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

              _createAdvancedSkySystem(THREE, Sky) {
                // Store THREE reference
                this._THREE = THREE;
                
                // console.debug('[SKY DEBUG] Creating Sky shader system...');
                
                // Create Sky shader
                this._sky = new Sky();
                this._sky.scale.setScalar(450000);
                this._scene.add(this._sky);
                
                // console.debug('[SKY DEBUG] Sky mesh created and added to scene');
                // console.debug('[SKY DEBUG] Sky scale:', this._sky.scale);
                // console.debug('[SKY DEBUG] Sky material:', this._sky.material);
                
                // Sky shader uniforms
                const skyUniforms = this._sky.material.uniforms;
                
                // Set initial daytime settings for visibility
                skyUniforms['turbidity'].value = 4.7;    // Default visible value
                skyUniforms['rayleigh'].value = 0.2;      // Default visible value
                skyUniforms['mieCoefficient'].value = 0.001;  // Default visible value
                skyUniforms['mieDirectionalG'].value = 0.9;   // Default visible value
                
                // console.debug('[SKY DEBUG] Initial sky shader uniforms:');
                // console.debug('  - turbidity:', skyUniforms['turbidity'].value);
                // console.debug('  - rayleigh:', skyUniforms['rayleigh'].value);
                // console.debug('  - mieCoefficient:', skyUniforms['mieCoefficient'].value);
                // console.debug('  - mieDirectionalG:', skyUniforms['mieDirectionalG'].value);
                
                // Create sun/moon light with default intensity
                this._sunLight = new THREE.DirectionalLight(0xffffff, 0.8);  // Default visible intensity
                this._sunLight.castShadow = true;
                this._sunLight.shadow.mapSize.width = 2048;
                this._sunLight.shadow.mapSize.height = 2048;
                this._sunLight.shadow.camera.near = 0.5;
                this._sunLight.shadow.camera.far = 500;
                // restrict shadow camera in smaller region can reduce aliasing and increase performance
                this._sunLight.shadow.camera.left = -75;
                this._sunLight.shadow.camera.right = 75;
                this._sunLight.shadow.camera.top = 75;
                this._sunLight.shadow.camera.bottom = -75;
                this._sunLight.shadow.bias = -0.001;
                this._sunLight.shadow.normalBias = 0.02;
                this._scene.add(this._sunLight);
                this._scene.add(this._sunLight.target);
                
                // Create moon light with increased intensity for better nighttime visibility
                this._moonLight = new THREE.DirectionalLight(0xff6633, 0.6); // Red color, slightly brighter
                this._moonLight.castShadow = true;
                this._moonLight.shadow.mapSize.width = 1024;
                this._moonLight.shadow.mapSize.height = 1024;
                this._moonLight.shadow.camera.near = 0.5;
                this._moonLight.shadow.camera.far = 500;
                this._moonLight.shadow.camera.left = -100;
                this._moonLight.shadow.camera.right = 100;
                this._moonLight.shadow.camera.top = 100;
                this._moonLight.shadow.camera.bottom = -100;
                this._moonLight.visible = false;
                this._scene.add(this._moonLight);
                this._scene.add(this._moonLight.target);
                
                // Store sun position for Sky shader
                this._sunPosition = new THREE.Vector3();
                
                // Create moon sphere
                this._createMoon(THREE);
                
                // Create god rays effect
                this._createGodRays(THREE);
                
                // Create stars for night sky
                this._createStars(THREE);
                
                // Create cloud system
                this._createCloudSystem(THREE);
                
                // Initialize with day settings - set initial sun position for visibility
                const phi = (90 - 45) * Math.PI / 180; // 45 degrees elevation
                const theta = 180 * Math.PI / 180; // 180 degrees azimuth
                const sunX = Math.sin(phi) * Math.cos(theta);
                const sunY = Math.cos(phi);
                const sunZ = Math.sin(phi) * Math.sin(theta);
                this._sunPosition = new THREE.Vector3(sunX, sunY, sunZ);
                this._sky.material.uniforms['sunPosition'].value.copy(this._sunPosition);
                
                // Initialize with visible day settings
                this._updateSkySystem(0.25); // Mid-day for good visibility
              }
              
              _createMoon(THREE) {
                  // Create a texture loader
                  const textureLoader = new THREE.TextureLoader();
                  // Load your new moon texture
                  const moonTexture = textureLoader.load('/experiment/static/moon_texture.jpg');

                  // --- Make the moon giant ---
                  const moonGeometry = new THREE.SphereGeometry(25, 64, 64); // Increased radius from 8 to 25

                  // --- Create the blood moon material ---
                  const moonMaterial = new THREE.MeshStandardMaterial({
                      map: moonTexture,                   // Apply the surface texture
                      emissiveMap: moonTexture,           // Make the texture itself glow
                      color: 0xff6633,                    // Tint the texture with a blood-orange color
                      emissive: 0xdd5522,                 // Set the glow color to a deep red
                      emissiveIntensity: 0.9,             // Adjust glow intensity
                      roughness: 1.0,                     // Keep it matte
                      metalness: 0.0
                  });

                  // Create moon mesh
                  this._moonMesh = new THREE.Mesh(moonGeometry, moonMaterial);
                  this._moonMesh.castShadow = false;
                  this._moonMesh.receiveShadow = false;

                  // --- Update the surrounding glow to be red ---
                  const moonGlowGeometry = new THREE.SphereGeometry(30, 32, 32); // Increased size
                  const moonGlowMaterial = new THREE.MeshBasicMaterial({
                      color: 0xdd5522,                    // Match the red glow color
                      transparent: true,
                      opacity: 0.1,
                      side: THREE.BackSide
                  });
                  this._moonGlow = new THREE.Mesh(moonGlowGeometry, moonGlowMaterial);
                  this._moonMesh.add(this._moonGlow);

                  // Initially hide moon
                  this._moonMesh.visible = false;
                  this._scene.add(this._moonMesh);

                  console.debug('[MOON] Giant blood moon created and added to scene');
              }
              
              _createGodRays(THREE) {
                  // Create a group to hold all the light beams
                  this._godRayGroup = new THREE.Group();
                  this._godRayGroup.name = 'godRays';
                  this._godRays = [];
                  const godRayCount = 12; // Fewer, more distinct beams look better

                  // A soft, circular texture for the beams
                  const canvas = document.createElement('canvas');
                  canvas.width = 128;
                  canvas.height = 128;
                  const ctx = canvas.getContext('2d');
                  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
                  gradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
                  gradient.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
                  ctx.fillStyle = gradient;
                  ctx.fillRect(0, 0, 128, 128);
                  const beamTexture = new THREE.CanvasTexture(canvas);

                  for (let i = 0; i < godRayCount; i++) {
                      // --- Use a Cylinder to create a 3D beam instead of a 2D plane ---
                      const rayLength = 350 + Math.random() * 100;
                      const rayWidth = 4 + Math.random() * 3;
                      const beamGeometry = new THREE.CylinderGeometry(rayWidth, rayWidth * 0.5, rayLength, 16, 1, true); // Open-ended cylinder

                      const beamMaterial = new THREE.MeshBasicMaterial({
                        map: beamTexture,
                        color: 0xffffff,
                        transparent: true,
                        opacity: 0.1 + Math.random() * 0.05, // Subtle opacity
                        blending: THREE.AdditiveBlending,
                        side: THREE.DoubleSide,
                        depthWrite: false,
                      });

                      const beam = new THREE.Mesh(beamGeometry, beamMaterial);

                      // --- Position and rotate each beam to point outward from the center ---
                      const angle = Math.random() * Math.PI * 2;
                      const spread = Math.random() * 0.1; // How far the beams spread out

                      // Position the beam slightly away from the center
                      beam.position.set(
                          Math.sin(angle) * spread * 150,
                          0,
                          Math.cos(angle) * spread * 150
                      );

                      // Point the cylinder outward
                      beam.lookAt(new THREE.Vector3(Math.sin(angle), spread, Math.cos(angle)));

                      // Store user data for animation
                      beam.userData = {
                          originalOpacity: beamMaterial.opacity,
                          phase: Math.random() * Math.PI * 2,
                          speed: 0.3 + Math.random() * 0.4
                      };

                      this._godRays.push(beam);
                      this._godRayGroup.add(beam);
                  }

                  this._godRayIntensity = 1.0;
                  this._godRayGroup.visible = true;
                  this._scene.add(this._godRayGroup);

                  console.debug('[GOD RAYS] Volumetric god rays system rebuilt with 3D beams');
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
              
              _updateSkySystem(phase) {
                if (!this._sky || !this._sunLight || !this._moonLight) {
                  console.warn('[SKY DEBUG] Missing sky components:', {
                    sky: !!this._sky,
                    sunLight: !!this._sunLight,
                    moonLight: !!this._moonLight
                  });
                  return;
                }
                
                // console.debug('[SKY DEBUG] Updating sky system for phase:', phase);
                
                // Dynamic sun/moon positioning based on game time
                // Phase: 0 = day (noon), 0.5 = transition, 1 = night (midnight)
                
                // Calculate time-based position
                // During DAY phase (0-0.5): sun moves from east to west
                // During NIGHT phase (0.5-1): moon moves from east to west
                
                let sunElevation, sunAzimuth;
                let moonElevation, moonAzimuth;
                
                const sunDistance = 400;
                const moonDistance = 400;
                
                if (phase <= 0.5) {
                    // DAY phase: sun is visible and moving
                    // Map phase 0-0.5 to sun movement from sunrise to sunset
                    const dayProgress = phase * 2; // 0 to 1 during day
                    
                    // Sun azimuth: moves from east (90°) through south (180°) to west (270°)
                    sunAzimuth = (90 + dayProgress * 180) * Math.PI / 180;
                    
                    // Sun elevation: rises from horizon, peaks at noon, sets at horizon
                    // Using a sine curve for smooth arc, max elevation 60° (user prefers lower)
                    const maxElevation = 60 * Math.PI / 180;
                    sunElevation = Math.sin(dayProgress * Math.PI) * maxElevation;
                    
                    // Keep sun at minimum 5° above horizon during most of day
                    if (dayProgress > 0.1 && dayProgress < 0.9) {
                        sunElevation = Math.max(sunElevation, 5 * Math.PI / 180);
                    }
                    
                    // Moon is below horizon during day
                    moonElevation = -10 * Math.PI / 180;
                    moonAzimuth = (sunAzimuth + Math.PI) % (2 * Math.PI); // Opposite side
                } else {
                    // NIGHT phase: moon is visible and moving
                    // Map phase 0.5-1 to moon movement from moonrise to moonset
                    const nightProgress = (phase - 0.5) * 2; // 0 to 1 during night
                    
                    // Moon azimuth: moves from east (90°) through south (180°) to west (270°)
                    moonAzimuth = (90 + nightProgress * 180) * Math.PI / 180;
                    
                    // Moon elevation: similar arc to sun but slightly lower max
                    const maxMoonElevation = 25 * Math.PI / 180; // 25 degrees max
                    moonElevation = Math.sin(nightProgress * Math.PI) * maxMoonElevation;
                    
                    // Keep moon at minimum 5° above horizon during most of night
                    if (nightProgress > 0.1 && nightProgress < 0.9) {
                        moonElevation = Math.max(moonElevation, 5 * Math.PI / 180);
                    }
                    
                    // Sun is below horizon during night
                    sunElevation = -10 * Math.PI / 180;
                    sunAzimuth = (moonAzimuth + Math.PI) % (2 * Math.PI); // Opposite side
                }
                
                // console.debug('[SKY DEBUG] Dynamic sun/moon calculations:', {
                //   phase,
                //   sunElevation: sunElevation * 180 / Math.PI,
                //   sunAzimuth: sunAzimuth * 180 / Math.PI,
                //   moonElevation: moonElevation * 180 / Math.PI,
                //   moonAzimuth: moonAzimuth * 180 / Math.PI
                // });
                
                // Calculate sun position
                const sunX = sunDistance * Math.sin(sunAzimuth) * Math.cos(sunElevation);
                const sunY = sunDistance * Math.sin(sunElevation);
                const sunZ = sunDistance * Math.cos(sunAzimuth) * Math.cos(sunElevation);
                
                this._sunPosition.set(sunX, sunY, sunZ);
                
                // console.debug('[SKY DEBUG] Sun position:', {
                //   x: sunX,
                //   y: sunY,
                //   z: sunZ,
                //   sunPosition: this._sunPosition
                // });
                
                // Update Sky shader sun position
                this._sky.material.uniforms['sunPosition'].value.copy(this._sunPosition);
                
                console.debug('[SKY DEBUG] Sky shader sunPosition uniform:',
                  this._sky.material.uniforms['sunPosition'].value);
                // Sun mesh positioning removed - using sky shader for visual representation
                
                
                // Position sun light
                this._sunLight.position.copy(this._sunPosition);
                this._sunLight.target.position.set(0, 0, 0);
                
                // Dynamic sun intensity based on elevation
                const sunIntensity = sunElevation > 0 ? Math.max(0, Math.sin(sunElevation)) : 0;
                this._sunLight.intensity = sunIntensity * 1.2;  // User preference: 3.0 at peak
                this._sunLight.visible = sunElevation > 0;
                
                // Adjust sun color based on elevation (redder at sunrise/sunset)
                const sunColorTemp = sunElevation < (10 * Math.PI / 180) ?
                    new this._THREE.Color(0xffaa66) : // Warm orange near horizon
                    new this._THREE.Color(0xffffff);  // White when higher
                this._sunLight.color = sunColorTemp;
                
                // Calculate moon position
                const moonX = moonDistance * Math.sin(moonAzimuth) * Math.cos(moonElevation);
                const moonY = moonDistance * Math.sin(moonElevation);
                const moonZ = moonDistance * Math.cos(moonAzimuth) * Math.cos(moonElevation);
                
                // Position moon mesh
                if (this._moonMesh) {
                  this._moonMesh.position.set(moonX, moonY, moonZ);
                  this._moonMesh.visible = moonElevation > 0;
                  
                  // Scale moon based on elevation (atmospheric magnification effect)
                  const moonScale = 1 + Math.max(0, (1 - Math.abs(moonElevation) / (10 * Math.PI / 180)) * 0.3);
                  this._moonMesh.scale.setScalar(moonScale);
                  
                  // Update moon glow intensity based on elevation
                  if (this._moonGlow && this._moonGlow.material) {
                    this._moonGlow.material.opacity = moonElevation > 0 ?
                      0.15 * Math.max(0, Math.sin(moonElevation)) : 0;
                  }
                }
                
                // Position moon light
                this._moonLight.position.set(moonX, moonY, moonZ);
                this._moonLight.target.position.set(0, 0, 0);
                this._moonLight.visible = moonElevation > 0;
                this._moonLight.intensity = moonElevation > 0 ? 0.4 : 0;  // Increased from 0.4 to 1.0 for better nighttime visibility
                
                // Update sky parameters based on time of day
                const skyUniforms = this._sky.material.uniforms;
                
                // Smooth transition between day and night sky parameters
                if (phase <= 0.5) {
                    // Day phase - use user's preferred values with slight transitions at dawn/dusk
                    const dayProgress = phase * 2;
                    
                    if (dayProgress < 0.1 || dayProgress > 0.9) {
                        // Dawn/dusk transition
                        const transitionFactor = dayProgress < 0.1 ? dayProgress * 10 : (1 - dayProgress) * 10;
                        skyUniforms['turbidity'].value = 10 - (10 - 0.6) * transitionFactor;
                        skyUniforms['rayleigh'].value = 0.1 + (1.9 - 0.1) * transitionFactor;
                        skyUniforms['mieCoefficient'].value = 0.005 + (0.010 - 0.005) * transitionFactor;
                        skyUniforms['mieDirectionalG'].value = 0.7 + (1.0 - 0.7) * transitionFactor;
                        console.debug('[SKY DEBUG] Applied DAWN/DUSK transition parameters');
                    } else {

                        skyUniforms['turbidity'].value = 4.7;    // Default visible value
                        skyUniforms['rayleigh'].value = 0.2;      // Default visible value
                        skyUniforms['mieCoefficient'].value = 0.001;  // Default visible value
                        skyUniforms['mieDirectionalG'].value = 0.9;   // Default visible value
                        
                        // Full day - user's preferred values
                        // skyUniforms['turbidity'].value = 0.6;      // User preference
                        // skyUniforms['rayleigh'].value = 1.9;       // User preference
                        // skyUniforms['mieCoefficient'].value = 0.010;  // User preference
                        // skyUniforms['mieDirectionalG'].value = 1.0;   // User preference
                        console.debug('[SKY DEBUG] Applied DAY sky parameters (user preferences)');
                    }
                } else {
                    // Night phase
                    const nightProgress = (phase - 0.5) * 2;
                    
                    if (nightProgress < 0.1 || nightProgress > 0.9) {
                        // Twilight transition
                        const transitionFactor = nightProgress < 0.1 ? (1 - nightProgress * 10) : nightProgress * 10;
                        skyUniforms['turbidity'].value = 0.6 + (10 - 0.6) * transitionFactor;
                        skyUniforms['rayleigh'].value = 1.9 - (1.9 - 0.1) * transitionFactor;
                        skyUniforms['mieCoefficient'].value = 0.010 - (0.010 - 0.005) * transitionFactor;
                        skyUniforms['mieDirectionalG'].value = 1.0 - (1.0 - 0.7) * transitionFactor;
                        console.debug('[SKY DEBUG] Applied TWILIGHT transition parameters');
                    } else {
                        // Full night
                        skyUniforms['turbidity'].value = 10;
                        skyUniforms['rayleigh'].value = 0.1;
                        skyUniforms['mieCoefficient'].value = 0.005;
                        skyUniforms['mieDirectionalG'].value = 0.7;
                        console.debug('[SKY DEBUG] Applied NIGHT sky parameters');
                    }
                }
                
                console.debug('[SKY DEBUG] Updated sky uniforms:', {
                  turbidity: skyUniforms['turbidity'].value,
                  rayleigh: skyUniforms['rayleigh'].value,
                  mieCoefficient: skyUniforms['mieCoefficient'].value,
                  mieDirectionalG: skyUniforms['mieDirectionalG'].value
                });
                
                // Update stars visibility based on actual darkness
                if (this._starsMaterial) {
                    // Stars visible during night phase
                    if (phase > 0.5) {
                        const nightProgress = (phase - 0.5) * 2;
                        // Fade in/out at twilight
                        if (nightProgress < 0.1) {
                            this._starsMaterial.uniforms.phase.value = nightProgress * 10;
                        } else if (nightProgress > 0.9) {
                            this._starsMaterial.uniforms.phase.value = (1 - nightProgress) * 10;
                        } else {
                            this._starsMaterial.uniforms.phase.value = 1;
                        }
                    } else {
                        this._starsMaterial.uniforms.phase.value = 0;
                    }
                }
                
                // Update clouds
                if (this._clouds) {
                    this._clouds.forEach(cloud => {
                        if (!cloud || !cloud.material) return;
                        // User preference: cloud opacity 0.3 (enabled)
                        cloud.material.opacity = 0.3;  // User preference: enabled
                    });
                }
                
                // Update god rays position and visibility
                if (this._godRayGroup && this._godRays) {
                    let godRayVisible = false;
                    let godRayPosition = null;
                    let godRayIntensity = 0;
                    let godRayColor = 0xffeeaa; // Default warm yellow
                    
                    // Determine if god rays should be visible based on sun/moon position
                    if (phase <= 0.5 && sunElevation > 0) {
                        // Day time - show god rays from sun
                        const dayProgress = phase * 2;
                        
                        // God rays are most visible during sunrise/sunset (low sun angles)
                        // and less visible at noon
                        if (sunElevation < 15 * Math.PI / 180) {
                            // Strong god rays at sunrise/sunset
                            godRayIntensity = 1.0;
                            godRayColor = 0xffaa66; // Warm orange for golden hour
                        } else if (sunElevation < 30 * Math.PI / 180) {
                            // Moderate god rays during morning/evening
                            godRayIntensity = 0.6;
                            godRayColor = 0xffddaa; // Warm yellow
                        } else {
                            // Subtle god rays at noon
                            godRayIntensity = 0.3;
                            godRayColor = 0xffffff; // White
                        }
                        
                        godRayVisible = true;
                        godRayPosition = this._sunPosition.clone();
                    } else if (phase > 0.5 && moonElevation > 0) {
                        // Night time - show god rays from moon
                        godRayIntensity = 0.4; // Softer for moonlight
                        godRayColor = 0xaaccff; // Cool blue-white for moon
                        godRayVisible = true;
                        
                        // Use moon position
                        godRayPosition = this._moonMesh.position.clone();
                    }
                    
                    // Apply god ray settings
                    this._godRayGroup.visible = godRayVisible && this._godRayIntensity > 0;
                    
                    if (godRayVisible && godRayPosition) {
                        // Position god rays at light source
                        this._godRayGroup.position.copy(godRayPosition);
                        
                        // Point rays toward the ground/origin
                        this._godRayGroup.lookAt(0, 0, 0);
                        
                        // Update ray colors and opacity based on intensity
                        this._godRays.forEach((ray, index) => {
                            if (ray.material) {
                                ray.material.color.setHex(godRayColor);
                                // Apply both scene intensity and user-controlled intensity
                                const finalOpacity = ray.userData.originalOpacity * godRayIntensity * this._godRayIntensity;
                                ray.material.opacity = finalOpacity;
                            }
                        });
                        
                        // console.debug('[GOD RAYS] Updated - Visible:', godRayVisible,
                        //            'Intensity:', godRayIntensity * this._godRayIntensity,
                        //            'Position:', godRayPosition);
                    }
                }
              }
              
              // Method to set god ray intensity (for external control)
              setGodRayIntensity(intensity) {
                this._godRayIntensity = Math.max(0, Math.min(2, intensity)); // Clamp between 0 and 2
                console.debug('[GOD RAYS] Intensity set to:', this._godRayIntensity);
              }
              
              _createCloudSystem(THREE) {
                this._clouds = [];
                const cloudCount = 8;
                
                // Create a procedural cloud texture with soft, feathered edges
                const createCloudTexture = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = 512;
                    canvas.height = 512;
                    const ctx = canvas.getContext('2d');
                    
                    // Clear canvas to transparent
                    ctx.clearRect(0, 0, 512, 512);
                    
                    // Create multiple overlapping soft circles to form cloud shape
                    const drawCloudBlob = (x, y, radius, opacity) => {
                        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
                        gradient.addColorStop(0, `rgba(255, 255, 255, ${opacity})`);
                        gradient.addColorStop(0.4, `rgba(255, 255, 255, ${opacity * 0.7})`);
                        gradient.addColorStop(0.7, `rgba(255, 255, 255, ${opacity * 0.3})`);
                        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
                        
                        ctx.fillStyle = gradient;
                        ctx.beginPath();
                        ctx.arc(x, y, radius, 0, Math.PI * 2);
                        ctx.fill();
                    };
                    
                    // Draw multiple overlapping blobs to create organic cloud shape
                    // Center blob
                    drawCloudBlob(256, 256, 180, 0.9);
                    // Surrounding blobs for irregular shape
                    drawCloudBlob(200, 220, 140, 0.8);
                    drawCloudBlob(320, 240, 150, 0.85);
                    drawCloudBlob(240, 300, 120, 0.75);
                    drawCloudBlob(300, 200, 130, 0.7);
                    // Smaller detail blobs
                    drawCloudBlob(180, 280, 90, 0.6);
                    drawCloudBlob(340, 280, 100, 0.65);
                    
                    const texture = new THREE.CanvasTexture(canvas);
                    texture.needsUpdate = true;
                    return texture;
                };
                
                // Create cloud geometry for each cloud (don't share geometry or texture)
                for (let i = 0; i < cloudCount; i++) {
                    // Create a unique cloud texture for each cloud for variety
                    const cloudTexture = createCloudTexture();
                    
                    // Use a simple plane geometry (no need for segments since we're using texture)
                    const cloudGeometry = new THREE.PlaneGeometry(60, 40);
                    
                    // Create material with cloud texture and proper transparency
                    const cloudMaterial = new THREE.MeshBasicMaterial({
                        map: cloudTexture,
                        transparent: true,
                        opacity: 0.3,
                        side: THREE.DoubleSide,
                        depthWrite: false,
                        alphaTest: 0.01 // Discard fully transparent pixels
                    });
                    
                    const cloud = new THREE.Mesh(cloudGeometry, cloudMaterial);
                    
                    // Validate cloud was created successfully
                    if (!cloud || !cloud.position) {
                        console.error(`[CLOUD ERROR] Failed to create cloud ${i}`);
                        continue;
                    }
                    
                    // Random position in sky
                    const angle = (i / cloudCount) * Math.PI * 2 + Math.random() * 0.5;
                    const radius = 200 + Math.random() * 100;
                    const height = 80 + Math.random() * 40;
                    
                    cloud.position.set(
                        Math.cos(angle) * radius,
                        height,
                        Math.sin(angle) * radius
                    );
                    
                    // Random rotation for variety
                    cloud.rotation.x = -Math.PI / 2 + Math.random() * 0.2;
                    cloud.rotation.z = Math.random() * Math.PI;
                    
                    // Random scale for variety
                    const scale = 0.8 + Math.random() * 0.4;
                    cloud.scale.set(scale, scale, scale);
                    
                    // Store initial position for animation
                    cloud.userData = {
                        initialAngle: angle,
                        radius: radius,
                        height: height,
                        speed: 0.0001 * 1.0  // Cloud speed multiplier: 1.0
                    };
                    
                    this._scene.add(cloud);
                    this._clouds.push(cloud);
                }
                
                // Disable clouds by default
                this._clouds.forEach(cloud => {
                    if (cloud) cloud.visible = false;
                });
                
                console.debug('[CLOUDS] Cloud system created with soft, feathered textures (disabled by default)');
              }

              _createAdvancedLighting(THREE) {
                // Increase ambient light for better, softer fill lighting
                const ambientLight = new THREE.AmbientLight(0x4a4a3a, 0.5);  // Increased from 0.1
                this._ambientLight = ambientLight; // Store reference for external access
                ambientLight.name = 'ambientLight';
                this._scene.add(ambientLight);

                // Rim light for depth - warm orange glow
                const rimLight = new THREE.DirectionalLight(0xaa6633, 0); // Set intensity to 0 to disable
                rimLight.position.set(-20, 10, -30);
                this._scene.add(rimLight);

                // Atmospheric hemisphere light - village sky
                const hemiLight = new THREE.HemisphereLight(0x6a7a9a, 0x3a2a1a, 0.5); // Increased from 0.3
                this._scene.add(hemiLight);

                // Ground fill light - warm village glow
                const fillLight = new THREE.DirectionalLight(0x5a4a3a, 0.15); // Warm fill
                fillLight.position.set(0, -1, 0);
                this._scene.add(fillLight);

                // Store references for phase updates
                this._rimLight = rimLight;
                this._hemiLight = hemiLight;
                this._fillLight = fillLight;

                // Create a spotlight for night actions - warm torchlight
                const spotLight = new THREE.SpotLight(0xffaa66, 0.8, 50, Math.PI / 3, 0.8, 2);
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
                // Create floating mystical particles - more for misty effect
                const particleCount = 300; // Increased for denser atmosphere
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
                  
                  // Mystical colors (blues, purples, greys) - more muted
                  const hue = Math.random() * 0.2 + 0.55; // 0.55-0.75 range (blue-purple)
                  const color = new THREE.Color().setHSL(hue, 0.3, 0.4); // Less saturated, darker
                  colors[i3] = color.r;
                  colors[i3 + 1] = color.g;
                  colors[i3 + 2] = color.b;
                  
                  sizes[i] = Math.random() * 1.5 + 0.3; // Smaller particles
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
                      
                      gl_FragColor = vec4(vColor, alpha * 0.3); // More subtle particles for mist
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

                // Bloom pass with user's preferred settings
                const bloomPass = new UnrealBloomPass(
                  new THREE.Vector2(this._width, this._height),
                  0.15,  // strength - User preference: 0.16
                  0.4,  // radius - User preference: 0.43
                  0.85   // threshold - User preference: 0.16
                );
                this._composer.addPass(bloomPass);

                // Film grain for atmosphere - subtle for quality
                const filmPass = new FilmPass(
                  0.15,  // noise intensity - subtle grain for texture
                  0.1,   // scanline intensity - minimal lines
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
                        // Night - gentle blue tint for moonlight
                        color.rgb = mix(color.rgb, color.rgb * vec3(0.85, 0.9, 1.15), 0.3);
                        color.rgb = pow(color.rgb, vec3(1.05));
                      } else {
                        // Day - slight warmth enhancement
                        color.rgb = mix(color.rgb, color.rgb * vec3(1.05, 1.0, 0.95), 0.2);
                      }
                      
                      // Add subtle vignette for depth
                      vec2 center = vec2(0.5, 0.5);
                      float dist = distance(vUv, center);
                      float vignette = 1.0 - smoothstep(0.4, 1.0, dist);
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
                this._renderer = this._threejs; // Store renderer reference for external access
                this._cloudSystem = this._clouds; // Store cloud system reference for external access
              }

              _LoadModels(THREE, FBXLoader, SkeletonUtils, CSS2DObject) {
                this._playerObjects = new Map();
                this._playerGroup = new THREE.Group();
                this._playerGroup.name = 'playerGroup';

                // Load the island model instead of creating a simple ground
                const radius = 15;
                
                // Load island FBX model with textures
                this._loadIslandModel(THREE, FBXLoader);

                // Call the new function to load the town background
                this._loadTownModel(THREE, FBXLoader);

                this._load_ground(THREE, this._EXRLoader);

                // Keep mystical circle patterns but adjust position if needed
                // this._createMysticalCircles(THREE, radius);

                this._scene.add(this._playerGroup);

                // Store references for later use
                this._THREE = THREE;
                this._CSS2DObject = CSS2DObject;
                
                // Cache Map objects
                this._modelCache = new Map();
                this._animationCache = new Map();
                
                // Role mapping object
                this._roleToDirectory = {
                  'Werewolf': 'werewolf',
                  'Doctor': 'doctor',
                  'Seer': 'seer',
                  'Villager': 'villager',
                  'Unknown': 'villager'
                };
                
                // Animation names array
                this._animationNames = ['Idle', 'Talking', 'Pointing', 'Victory', 'Defeated', 'Dying'];
                
                // Animation file variants mapping
                this._animationFileVariants = {
                  'Idle': ['Idle.fbx', 'Standing Idle.fbx', 'Neutral Idle.fbx'],
                  'Talking': ['Talking.fbx'],
                  'Pointing': ['Pointing.fbx'],
                  'Victory': ['Victory.fbx'],
                  'Defeated': ['Defeated.fbx'],
                  'Dying': ['Dying.fbx']
                };
                
                // Create particle system for atmosphere
                this._createParticleSystem(THREE);
                
                // Create campfire at center
                this._createCampfire(THREE);
                
                // Frame the empty group initially with better camera positioning
                this._camera.position.set(25, 30, 35);
                this._controls.target.set(0, 8, 0);
                this._controls.enableDamping = true;
                this._controls.dampingFactor = 0.05;
                this._controls.minDistance = 20;
                this._controls.maxDistance = 80;
                this._controls.maxPolarAngle = Math.PI * 0.6;
                this._controls.update();
              }
              
              _createCampfire(THREE) {
                console.debug('[CAMPFIRE] Creating campfire at scene center');
                
                // Set texture path for VolumetricFire
                if (this._VolumetricFire) {
                  this._VolumetricFire.texturePath = '/experiment/static/volumetric_fire/textures/';
                }
                
                // Create campfire group
                const campfireGroup = new THREE.Group();
                campfireGroup.name = 'campfire';
                
                // Create fire using VolumetricFire - doubled in size
                const fireWidth = 5.0;   // 2x larger (was 2.5)
                const fireHeight = 7.0;  // 2x larger (was 3.5)
                const fireDepth = 5.0;   // 2x larger (was 2.5)
                const sliceSpacing = 0.5;
                
                if (this._VolumetricFire) {
                  this._fire = new this._VolumetricFire(
                    fireWidth,
                    fireHeight,
                    fireDepth,
                    sliceSpacing,
                    this._camera
                  );
                  this._fire.mesh.position.set(0, fireHeight / 2, 0);
                  campfireGroup.add(this._fire.mesh);
                  console.debug('[CAMPFIRE] VolumetricFire created');
                } else {
                  console.warn('[CAMPFIRE] VolumetricFire not available, skipping fire effect');
                }
                
                // Add point light for fire glow
                const fireLight = new THREE.PointLight(0xff6633, 2.5, 25);
                fireLight.position.set(0, 1.5, 0);
                fireLight.castShadow = true;
                fireLight.shadow.mapSize.width = 512;
                fireLight.shadow.mapSize.height = 512;
                campfireGroup.add(fireLight);
                this._fireLight = fireLight;
                
                // Create rock circle around fire base
                const rockCount = 8;
                const rockRadius = 2.2;
                const textureLoader = new THREE.TextureLoader();
                
                for (let i = 0; i < rockCount; i++) {
                  const angle = (i / rockCount) * Math.PI * 2;
                  const x = Math.cos(angle) * rockRadius;
                  const z = Math.sin(angle) * rockRadius;
                  
                  // Vary rock sizes
                  const rockSize = 0.4 + Math.random() * 0.3;
                  const rockGeometry = new THREE.DodecahedronGeometry(rockSize, 0);
                  
                  // Rock material - dark grey with rough texture
                  const rockMaterial = new THREE.MeshStandardMaterial({
                    color: 0x3a3a3a,
                    roughness: 0.95,
                    metalness: 0.1,
                    flatShading: true
                  });
                  
                  const rock = new THREE.Mesh(rockGeometry, rockMaterial);
                  rock.position.set(x, rockSize * 0.3, z);
                  
                  // Random rotation for variety
                  rock.rotation.set(
                    Math.random() * Math.PI,
                    Math.random() * Math.PI,
                    Math.random() * Math.PI
                  );
                  
                  rock.castShadow = true;
                  rock.receiveShadow = true;
                  campfireGroup.add(rock);
                }
                
                // Create logs arranged in a teepee style
                const logCount = 6;
                const logLength = 2.0;
                const logRadius = 0.12;
                
                for (let i = 0; i < logCount; i++) {
                  const angle = (i / logCount) * Math.PI * 2;
                  
                  // Log geometry - cylinder
                  const logGeometry = new THREE.CylinderGeometry(
                    logRadius,
                    logRadius * 0.9,
                    logLength,
                    8
                  );
                  
                  // Wood material - brown with bark texture
                  const logMaterial = new THREE.MeshStandardMaterial({
                    color: 0x4a3520,
                    roughness: 0.9,
                    metalness: 0.0
                  });
                  
                  const log = new THREE.Mesh(logGeometry, logMaterial);
                  
                  // Position logs leaning inward
                  const leanRadius = 0.8;
                  const x = Math.cos(angle) * leanRadius;
                  const z = Math.sin(angle) * leanRadius;
                  
                  log.position.set(x, logLength / 2 - 0.3, z);
                  
                  // Rotate to lean inward toward center
                  log.rotation.z = Math.PI / 6; // Lean angle
                  log.rotation.y = angle + Math.PI / 2; // Face center
                  
                  log.castShadow = true;
                  log.receiveShadow = true;
                  campfireGroup.add(log);
                }
                
                // Add some smaller kindling pieces at the base
                const kindlingCount = 12;
                for (let i = 0; i < kindlingCount; i++) {
                  const angle = Math.random() * Math.PI * 2;
                  const radius = Math.random() * 0.6;
                  
                  const kindlingGeometry = new THREE.CylinderGeometry(
                    0.03,
                    0.025,
                    0.4 + Math.random() * 0.3,
                    6
                  );
                  
                  const kindlingMaterial = new THREE.MeshStandardMaterial({
                    color: 0x3a2510,
                    roughness: 0.95,
                    metalness: 0.0
                  });
                  
                  const kindling = new THREE.Mesh(kindlingGeometry, kindlingMaterial);
                  kindling.position.set(
                    Math.cos(angle) * radius,
                    0.2,
                    Math.sin(angle) * radius
                  );
                  kindling.rotation.set(
                    Math.random() * 0.5,
                    Math.random() * Math.PI * 2,
                    Math.random() * 0.5
                  );
                  
                  kindling.castShadow = true;
                  kindling.receiveShadow = true;
                  campfireGroup.add(kindling);
                }
                
                // Position campfire at scene center
                campfireGroup.position.set(0, 0, 0);
                this._scene.add(campfireGroup);
                this._campfireGroup = campfireGroup;
                
                console.debug('[CAMPFIRE] Campfire scene created with rocks and logs');
              }

              _loadIslandModel(THREE, FBXLoader) {
                const textureLoader = new THREE.TextureLoader();
                
                // Load all textures
                const baseTexture = textureLoader.load('/experiment/static/werewolf/island/_0930062431_texture.png');
                const normalTexture = textureLoader.load('/experiment/static/werewolf/island/_0930062431_texture_normal.png');
                const metallicTexture = textureLoader.load('/experiment/static/werewolf/island/_0930062431_texture_metallic.png');
                const roughnessTexture = textureLoader.load('/experiment/static/werewolf/island/_0930062431_texture_roughness.png');
                
                // Configure textures
                [baseTexture, normalTexture, metallicTexture, roughnessTexture].forEach(texture => {
                  texture.encoding = THREE.sRGBEncoding;
                  texture.flipY = true;
                });
                
                // Load the FBX model
                const fbxLoader = new FBXLoader();
                fbxLoader.load(
                  '/experiment/static/werewolf/island/_0930062431_texture.fbx',
                  (fbx) => {
                    // Scale and position the island
                    fbx.scale.setScalar(0.02); // Much larger scale for visibility
                    fbx.position.y = -19.8; // Position at ground level
                    fbx.rotation.y = Math.PI / 8; // Slight rotation for better view
                    
                    // Apply textures to all meshes in the model
                    fbx.traverse((child) => {
                      if (child.isMesh) {
                        // Create PBR material with all textures
                        const material = new THREE.MeshStandardMaterial({
                          map: baseTexture,
                          normalMap: normalTexture,
                          normalScale: new THREE.Vector2(0.5, 0.5), // Reduced normal intensity
                          metalnessMap: metallicTexture,
                          roughnessMap: roughnessTexture,
                          metalness: 0.1, // Much less metallic to reduce brightness
                          roughness: 0.95, // More rough to scatter light
                          envMapIntensity: 0.2, // Reduced environment reflection
                          color: new THREE.Color(0.75, 0.75, 0.75), // Darken the base color
                          side: THREE.DoubleSide // Render both sides
                        });
                        
                        child.material = material;
                        child.castShadow = true;
                        child.receiveShadow = true;
                      }
                    });
                    
                    // Add to scene
                    this._scene.add(fbx);
                    
                    // Store reference if needed
                    this._islandModel = fbx;
                    
                    console.debug('Island model loaded successfully');
                  },
                  (progress) => {
                    console.debug('Loading island model:', (progress.loaded / progress.total * 100).toFixed(2) + '%');
                  },
                  (error) => {
                    console.error('Error loading island model:', error);
                    
                    // Fallback to simple ground if island fails to load
                    const groundGeometry = new THREE.CircleGeometry(20, 64);
                    const groundMaterial = new THREE.MeshStandardMaterial({
                      color: 0x1a1a2a,
                      roughness: 0.9,
                      metalness: 0.1,
                      transparent: true,
                      opacity: 0.95
                    });
                    
                    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
                    ground.rotation.x = -Math.PI / 2;
                    ground.position.y = -0.1;
                    ground.receiveShadow = true;
                    this._scene.add(ground);
                  }
                );
              }

              _loadTownModel(THREE, FBXLoader) {
                // The path to your town's FBX file.
                // Make sure this path is correct for your project structure.
                const townModelPath = '/experiment/static/werewolf/town/scene_v1.fbx'; // <-- REPLACE WITH YOUR FILE PATH

                console.debug(`[Town Loader] Attempting to load model from: ${townModelPath}`);

                this._fbxLoader.load(
                  townModelPath,
                  (fbx) => {
                    console.debug('[Town Loader] Model loaded successfully.');

                    // --- ADJUSTMENTS ---
                    // You can change these values to fit the town into your scene.

                    // 1. SCALING: Adjust the number to make the town bigger or smaller.
                    //    - 1.0 is original size.
                    //    - 0.1 would be 10% of the original size.
                    fbx.scale.setScalar(0.15);

                    // 2. POSITIONING: Change the X, Y, and Z values to move the town.
                    //    - Y is the vertical axis. You will likely need to adjust this
                    //      to make the town sit at the correct ground level.
                    fbx.position.set(0, 0, 0);

                    // 3. ROTATION: Adjust the rotation around the vertical (Y) axis.
                    //    - The value is in radians. Math.PI is 180 degrees.
                    //    - Math.PI / 2 is a 90-degree turn.
                    fbx.rotation.y = Math.PI / 2;


                    // --- SCENE INTEGRATION ---
                    // This loop goes through every part of your town model.
                    fbx.traverse((child) => {
                      if (child.isMesh) {
                        // This makes the town cast and receive shadows, which is crucial
                        // for making it look like it belongs in the scene.
                        child.castShadow = true;
                        child.receiveShadow = true;

                        child.material.normalMap = null;
                        child.material.metalnessMap = null;

                        // Optional: If your baked textures look too shiny, you can
                        // make them appear more matte with these lines.
                        if (child.material) {
                            child.material.roughness = 1.0;
                            child.material.metalness = 0.0;
                        }

                        
                      }
                    });

                    // Add the finished town model to the main scene.
                    this._scene.add(fbx);
                    
                    // Store a reference if you need to access it later.
                    this._townModel = fbx;
                  },
                  (progress) => {
                    // This will show the loading progress in the console.
                    console.debug('[Town Loader] Loading progress: ' + (progress.loaded / progress.total * 100).toFixed(2) + '%');
                  },
                  (error) => {
                    // This will log an error if the file can't be found or loaded.
                    console.error('[Town Loader] An error happened while loading the town model:', error);
                  }
                );
              }

              /**
               * Creates a realistic, rocky ground plane using PBR textures.
               * @param {object} THREE - The THREE.js library instance.
               * @param {object} EXRLoader - The loader for .exr texture files.
               */
              _load_ground(THREE, EXRLoader) {
                  console.debug('[Ground Loader] Creating realistic rocky terrain...');

                  // 1. Initialize two separate loaders for the different file types.
                  const textureLoader = new THREE.TextureLoader();
                  const exrLoader = new EXRLoader();

                  // 2. Load your specific PBR textures.
                  const colorTexture = textureLoader.load('/experiment/static/werewolf/ground/rocky_terrain_02_diff_1k.jpg');
                  const displacementTexture = textureLoader.load('/experiment/static/werewolf/ground/rocky_terrain_02_disp_1k.png');
                  
                  // --- Use the EXRLoader for the .exr files ---
                  const roughnessTexture = exrLoader.load('/experiment/static/werewolf/ground/rocky_terrain_02_rough_1k.exr');
                  const normalTexture = exrLoader.load('/experiment/static/werewolf/ground/rocky_terrain_02_nor_gl_1k.exr');

                  // 3. Configure all textures to repeat (tile) across the ground.
                  // A smaller repeat value like (8, 8) makes the rock features appear larger.
                  [colorTexture, roughnessTexture, displacementTexture, normalTexture].forEach(texture => {
                      texture.wrapS = THREE.RepeatWrapping;
                      texture.wrapT = THREE.RepeatWrapping;
                      texture.repeat.set(16, 16);
                  });

                  // 4. Define the ground's geometry. A high segment count is crucial for displacement.
                  const groundGeometry = new THREE.CircleGeometry(200, 128);

                  // 5. Define the material using your textures.
                  const groundMaterial = new THREE.MeshStandardMaterial({
                      map: colorTexture,            // from rocky_terrain_02_diff_1k.jpg
                      roughnessMap: roughnessTexture, // from rocky_terrain_02_rough_1k.exr
                      normalMap: normalTexture,     // from rocky_terrain_02_nor_gl_1k.exr
                      displacementMap: displacementTexture, // from rocky_terrain_02_disp_1k.png
                      displacementScale: 0.5,       // Adjust this value to make the terrain more or less hilly.
                  });

                  // 6. Create the final Mesh, rotate it, enable shadow reception, and add to the scene.
                  const ground = new THREE.Mesh(groundGeometry, groundMaterial);
                  ground.rotation.x = -Math.PI / 2; // Lay the plane flat.
                  ground.position.y = -0.75;
                  ground.receiveShadow = true;
                  this._scene.add(ground);

                  console.debug('[Ground Loader] Rocky terrain created and added to the scene.');
              }

              loadCharacterModel(role) {
                // Normalize the role using _roleToDirectory, defaulting to 'villager'
                const normalizedRole = this._roleToDirectory[role] || this._roleToDirectory['Villager'] || 'villager';
                
                // Check if model is already cached
                if (this._modelCache.has(normalizedRole)) {
                  return this._modelCache.get(normalizedRole);
                }
                
                // Create a new promise for loading the merged FBX file
                const modelPromise = new Promise((resolve, reject) => {
                  const modelPath = `/experiment/static/werewolf/models/${normalizedRole}/${normalizedRole}.fbx`;
                  
                  this._fbxLoader.load(
                    modelPath,
                    (fbx) => {
                      // On success: apply uniform scaling
                      fbx.scale.setScalar(0.05); // 3.5x larger than original 0.01
                      
                      // Correct the orientation - rotate 90 degrees to face forward
                    //   fbx.rotation.y = -Math.PI / 2;
                      
                      // Extract and store animations from the merged FBX
                      const animations = {};
                      if (fbx.animations && fbx.animations.length > 0) {
                        fbx.animations.forEach((clip) => {
                          // Skip animations that start with "Armature|" as they are duplicates
                          if (clip.name.startsWith('Armature|')) {
                            console.debug(`Skipping duplicate animation: ${clip.name}`);
                            return;
                          }
                          
                          // Map animation names based on common patterns
                          let animName = clip.name;
                          
                          // Try to match known animation names
                          if (animName.toLowerCase().includes('idle') || animName.toLowerCase().includes('standing')) {
                            animations['Idle'] = clip;
                            clip.name = 'Idle';
                          } else if (animName.toLowerCase().includes('talk')) {
                            animations['Talking'] = clip;
                            clip.name = 'Talking';
                          } else if (animName.toLowerCase().includes('point')) {
                            animations['Pointing'] = clip;
                            clip.name = 'Pointing';
                          } else if (animName.toLowerCase().includes('victory') || animName.toLowerCase().includes('win')) {
                            animations['Victory'] = clip;
                            clip.name = 'Victory';
                          } else if (animName.toLowerCase().includes('defeat') || animName.toLowerCase().includes('lose')) {
                            animations['Defeated'] = clip;
                            clip.name = 'Defeated';
                          } else if (animName.toLowerCase().includes('dying') || animName.toLowerCase().includes('death')) {
                            animations['Dying'] = clip;
                            clip.name = 'Dying';
                          } else {
                            // Store with original name as fallback
                            animations[animName] = clip;
                          }
                        });
                      }
                      
                      // Store animations in cache for this role
                      this._animationCache.set(normalizedRole, Promise.resolve(animations));
                      
                      // Store the original model in cache, cloning will happen per-player
                      resolve(fbx);
                    },
                    (progress) => {
                      // Progress callback (optional)
                      console.debug(`Loading merged model for ${normalizedRole}: ${(progress.loaded / progress.total * 100).toFixed(2)}%`);
                    },
                    (error) => {
                      // On error: reject with descriptive message
                      reject(new Error(`Failed to load merged model for role '${role}' (normalized: '${normalizedRole}'): ${error.message || error}`));
                    }
                  );
                });
                
                // Store this promise in cache immediately to dedupe concurrent calls
                this._modelCache.set(normalizedRole, modelPromise);
                
                // Return the promise
                return modelPromise;
              }

              loadCharacterAnimations(role) {
                // Normalize the role using _roleToDirectory, defaulting to 'villager'
                const normalizedRole = this._roleToDirectory[role] || this._roleToDirectory['Villager'] || 'villager';
                
                // Check if animations are already cached (they should be from loadCharacterModel)
                if (this._animationCache.has(normalizedRole)) {
                  return this._animationCache.get(normalizedRole);
                }
                
                // If not cached, it means the model hasn't been loaded yet
                // Return an empty promise that will be resolved when the model loads
                console.warn(`Animations for ${normalizedRole} not yet loaded. They will be extracted from the merged FBX.`);
                return Promise.resolve({});
              }

              // This method is no longer needed since animations are extracted from the merged FBX
              // Keeping it as a stub for compatibility
              _loadAnimationWithFallbacks(role, animationName) {
                console.debug(`_loadAnimationWithFallbacks called for ${role}/${animationName} - this should not happen with merged FBX files`);
                return Promise.resolve(null);
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
                const { orb, orbLight, glow, pedestal, container } = player;
                
                orb.material.emissiveIntensity = 1.;
                orbLight.intensity = 1.;
                glow.material.emissiveIntensity = 0.5;
                // Slight scale up animation
                container.scale.setScalar(1.1);
                pedestal.material.emissiveIntensity = 0.3;
              }

              updatePlayerStatus(playerName, player_info, status, threatLevel = 0, is_active = false) {
                const player = this._playerObjects.get(playerName);
                if (!player) return;

                const { orb, orbLight, model, glow, pedestal, container, mixer, animations, currentAction } = player;

                if (player_info && player_info.role !== 'Unknown' && player.playerUI) {
                    const roleElement = player.playerUI.element.querySelector('.player-role-3d');
                    if (roleElement) {
                        let roleDisplay = player_info.role;
                        let roleColor = '#00b894'; // Villager green
                        if (player_info.role === 'Werewolf') {
                            roleDisplay = `\u{1F43A} ${player_info.role}`; // 🐺 Wolf emoji
                            roleColor = '#e17055'; // Werewolf red
                        } else if (player_info.role === 'Doctor') {
                            roleDisplay = `\u{1FA7A} ${player_info.role}`; // 🩺 Stethoscope emoji
                            roleColor = '#6c5ce7'; // Doctor purple
                        } else if (player_info.role === 'Seer') {
                            roleDisplay = `\u{1F52E} ${player_info.role}`; // 🔮 Crystal Ball emoji
                            roleColor = '#fd79a8'; // Seer pink
                        } else if (player_info.role === 'Villager') {
                            roleDisplay = `\u{1F33E} ${player_info.role}`; //  🌾 emoji
                            roleColor = '#00b894'; // Villager green
                        }
                        roleElement.textContent = `${roleDisplay}`;
                        roleElement.style.color = roleColor;
                        roleElement.style.fontWeight = 'bold';
                    }
                }

                // Reset to default state - more muted
                orb.material.color.setHex(0x00aa88);
                orb.material.emissive.setHex(0x00aa88);
                orb.material.emissiveIntensity = 1.0;
                orb.material.opacity = 0.85;
                orb.visible = true;
                orbLight.color.setHex(0x00aa88);
                orbLight.intensity = 0.8;
                orbLight.visible = true;
                
                // For FBX models, we don't change material colors directly
                // Instead, we rely on animations and visual effects
                
                glow.material.color.setHex(0x00aa88);
                glow.material.emissive.setHex(0x00aa88);
                glow.material.emissiveIntensity = 0.15;
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
                
                // Handle animations based on status
                switch(status) {
                    case 'dead':
                        orb.visible = false;
                        orbLight.visible = false;
                        glow.visible = false;
                        pedestal.material.emissive.setHex(0x050505);
                        if (player.nameplate && player.nameplate.element) {
                            player.nameplate.element.style.transition = 'opacity 2s ease-out';
                            player.nameplate.element.style.opacity = '0.7';
                        }
                        player.isAlive = false;

                        // --- START OF NEW LOGIC ---
                        mixer.stopAllAction(); // IMPORTANT: Stop any other animations like 'Idle'.

                        const dyingAnimation = animations ? animations['Dying'] : null;
                        if (dyingAnimation) {
                            const deathMap = window.werewolfThreeJs.deathAnimationCompleted;
                            const action = mixer.clipAction(dyingAnimation);
                            action.setLoop(this._THREE.LoopOnce);
                            action.clampWhenFinished = true;

                            if (deathMap && !deathMap.has(playerName)) {
                                // This is the first time we're seeing this player die. Play the full animation.
                                console.debug(`[DEATH TRIGGER] Playing full death animation for ${playerName}.`);
                                action.reset().play();
                                deathMap.set(playerName, true); // Mark it as completed.
                            } else {
                                // Player is already dead. Force the model to the final frame of the animation.
                                action.play();
                                action.time = action.getClip().duration; // Jump to the end.
                            }
                        } else {
                            // Fallback for models without a 'Dying' animation.
                            container.rotation.x = 0.2;
                        }
                        break;
                    case 'werewolf':
                        glow.material.color.setHex(0xaa4444); // Muted red
                        glow.material.emissive.setHex(0xaa4444);
                        glow.material.emissiveIntensity = 0.2;
                        glow.visible = true;
                        pedestal.material.emissive.setHex(0x440000);
                        pedestal.material.emissiveIntensity = 0.2;
                        break;
                    case 'doctor':
                        glow.material.color.setHex(0x44aa44); // Muted green
                        glow.material.emissive.setHex(0x44aa44);
                        glow.material.emissiveIntensity = 0.2;
                        glow.visible = true;
                        pedestal.material.emissive.setHex(0x002200);
                        pedestal.material.emissiveIntensity = 0.1;
                        break;
                    case 'seer':
                        glow.material.color.setHex(0x6644aa); // Muted purple
                        glow.material.emissive.setHex(0x6644aa);
                        glow.material.emissiveIntensity = 0.2;
                        glow.visible = true;
                        pedestal.material.emissive.setHex(0x1A002A);
                        pedestal.material.emissiveIntensity = 0.1;
                        break;
                    default:
                        // Keep default state
                        break;
                }

                // Update threat level indicators - more muted colors
                if (threatLevel >= 1.0) { // DANGER
                    orb.material.color.setHex(0xaa4444); // Muted red
                    orb.material.emissive.setHex(0xaa4444);
                    orb.material.emissiveIntensity = 1.2;
                    orb.material.opacity = 0.8;
                    orbLight.color.setHex(0xaa4444);
                    orbLight.intensity = 0.6;
                    glow.material.color.setHex(0xaa4444);
                    glow.material.emissive.setHex(0xaa4444);
                    glow.material.emissiveIntensity = 0.2;
                } else if (threatLevel >= 0.5) { // UNEASY
                    orb.material.color.setHex(0xaaaa44); // Muted yellow
                    orb.material.emissive.setHex(0xaaaa44);
                    orb.material.emissiveIntensity = 1.1;
                    orb.material.opacity = 0.75;
                    orbLight.color.setHex(0xaaaa44);
                    orbLight.intensity = 0.5;
                    glow.material.color.setHex(0xaaaa44);
                    glow.material.emissive.setHex(0xaaaa44);
                    glow.material.emissiveIntensity = 0.18;
                } else { // SAFE
                    orb.material.color.setHex(0x44aa88); // Muted teal-green
                    orb.material.emissive.setHex(0x44aa88);
                    orb.material.emissiveIntensity = 0.8;
                    orb.material.opacity = 0.7;
                    orbLight.color.setHex(0x44aa88);
                    orbLight.intensity = 0.4;
                    glow.material.color.setHex(0x44aa88);
                    glow.material.emissive.setHex(0x44aa88);
                    glow.material.emissiveIntensity = 0.15;
                }
                
                // Ensure alive players without animation are in Idle
                if (player.isAlive && mixer && animations && animations['Idle'] && !currentAction) {
                    this.playAnimation(playerName, 'Idle');
                }
              }

              playAnimation(playerName, animationName, options = {}) {
                const player = this._playerObjects.get(playerName);
                if (!player || !player.model) return null;
                
                // CRITICAL: Check death animation completion FIRST, before any other checks
                const deathMap = window.werewolfThreeJs.deathAnimationCompleted;
                if (deathMap && deathMap.has(playerName)) {
                    // Player has already completed death animation, block ALL animations
                    console.debug(`[BLOCKED] Animation '${animationName}' blocked for ${playerName} - death animation already completed`);
                    return null;
                }
                
                // Check if player is dead and animation is not death-related
                if (!player.isAlive &&
                    !['Dying', 'Defeated', 'Victory'].includes(animationName)) {
                    console.debug(`[SKIP] Animation '${animationName}' skipped for dead player ${playerName}`);
                    return null;
                }
                
                const animations = player.animations;
                if (!animations || !animations[animationName]) return null;
                
                const mixer = player.mixer;
                if (!mixer) return null;
                
                // If this is a death animation, immediately mark it as completed
                if (animationName === 'Dying' || animationName === 'Defeated') {
                    if (deathMap) {
                        console.debug(`[DEATH START] Starting death animation '${animationName}' for ${playerName} - marking as completed`);
                        deathMap.set(playerName, true);
                    }
                }
                
                // Fade out current action if exists
                if (player.currentAction) {
                    player.currentAction.fadeOut(options.fadeOutDuration || 0.2);
                }
                
                // Create new action
                const action = mixer.clipAction(animations[animationName]);
                action.reset();
                
                // Auto-determine loop mode based on animation type
                // if (['Idle', 'Talking', 'Pointing'].includes(animationName)) {
                //     action.setLoop(this._THREE.LoopRepeat);
                //     action.clampWhenFinished = false;
                // } else if (['Victory', 'Defeated', 'Dying'].includes(animationName)) {
                //     // action.setLoop(this._THREE.LoopOnce);
                //     action.setLoop(this._THREE.LoopRepeat);
                //     action.clampWhenFinished = false;
                //     console.debug(`[DEATH CONFIG] Death animation '${animationName}' configured with LoopOnce and clampWhenFinished for ${playerName}`);
                // }
                action.setLoop(this._THREE.LoopRepeat);

                
                // Apply any custom options
                if (options.loop !== undefined) {
                    action.setLoop(options.loop);
                }
                if (options.clampWhenFinished !== undefined) {
                    action.clampWhenFinished = options.clampWhenFinished;
                }
                
                // Play with fade-in
                action.fadeIn(options.fadeInDuration || 0.2);
                action.play();
                
                // Update player's current action
                player.currentAction = action;
                
                return action;
              }

              displayPlayerBubble(playerName, message) {
                  const player = this._playerObjects.get(playerName);
                  if (!player || !player.playerUI) return;

                  const uiElement = player.playerUI.element;
                  const messageEl = uiElement.querySelector('.bubble-message');

                  // Update the message content
                  messageEl.innerHTML = message;

                  // Add the 'chat-active' class to trigger the CSS animation
                  uiElement.classList.add('chat-active');                  
              }

              triggerSpeakingAnimation(playerName) {
                const player = this._playerObjects.get(playerName);
                if (!player || !player.isAlive) return;

                // Use centralized animation method
                this.playAnimation(playerName, 'Talking', {
                    fadeInDuration: 0.2,
                    fadeOutDuration: 0.2
                });
                
                // Schedule return to idle
                setTimeout(() => {
                    this.playAnimation(playerName, 'Idle', {
                        fadeInDuration: 0.2
                    });
                }, 1800);

                // Also add visual sound wave effect
                const wave = this._createSoundWave(this._THREE);
                player.container.add(wave);

                // Add the wave to our animation manager array
                this._speakingAnimations.push({
                    mesh: wave,
                    startTime: performance.now(),
                    duration: 1800, // Animation duration in milliseconds
                });
              }

              triggerPointingAnimation(playerName, duration = 1200) {
                const player = this._playerObjects.get(playerName);
                if (!player || !player.isAlive) return;
                
                // Play pointing animation
                this.playAnimation(playerName, 'Pointing', {
                    fadeInDuration: 0.2,
                    fadeOutDuration: 0.2
                });
                
                // Return to idle after duration
                setTimeout(() => {
                    this.playAnimation(playerName, 'Idle', {
                        fadeInDuration: 0.2
                    });
                }, duration);
              }

              triggerVictoryAnimation(playerName) {
                const player = this._playerObjects.get(playerName);
                if (!player || !player.model) return;
                
                this.playAnimation(playerName, 'Victory');
              }

              triggerDefeatedAnimation(playerName) {
                const player = this._playerObjects.get(playerName);
                if (!player || !player.model) return;
                
                this.playAnimation(playerName, 'Defeated');
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
                
                // Calculate target phase value with smooth time progression
                // DAY phase: 0.0 (dawn) -> 0.25 (noon) -> 0.5 (dusk)
                // NIGHT phase: 0.5 (dusk) -> 0.75 (midnight) -> 1.0 (dawn)
                let targetPhase;
                
                if (normalizedPhase === 'NIGHT') {
                    // For night, start at 0.5 and progress to approach 1.0
                    // Add some time progression within the night phase
                    if (!this._nightStartTime) {
                        this._nightStartTime = Date.now();
                        this._dayStartTime = null;
                    }
                    const nightDuration = 30000; // 30 seconds for full night cycle
                    const nightElapsed = Date.now() - this._nightStartTime;
                    const nightProgress = Math.min(nightElapsed / nightDuration, 1.0);
                    targetPhase = 0.5 + nightProgress * 0.5; // 0.5 to 1.0
                } else {
                    // For day, start at 0.0 and progress to 0.5
                    // Add some time progression within the day phase
                    if (!this._dayStartTime) {
                        this._dayStartTime = Date.now();
                        this._nightStartTime = null;
                    }
                    const dayDuration = 30000; // 30 seconds for full day cycle
                    const dayElapsed = Date.now() - this._dayStartTime;
                    const dayProgress = Math.min(dayElapsed / dayDuration, 1.0);
                    targetPhase = dayProgress * 0.5; // 0.0 to 0.5
                }
                
                // Initialize transition system if not exists
                if (!this._phaseTransition) {
                    this._phaseTransition = {
                        current: targetPhase,
                        target: targetPhase,
                        speed: 0.02 // Smooth transition speed
                    };
                    // Immediately set to target on first call
                    this._updateSceneForPhase(targetPhase);
                } else {
                    // Update target for smooth transition
                    this._phaseTransition.target = targetPhase;
                }
              }
              
              _updateSceneForPhase(phaseValue) {
                const THREE = this._THREE;
                
                // Update renderer tone mapping for day/night mood
                if (this._threejs) {
                    this._threejs.toneMappingExposure = 0.5 + (0.3 - phaseValue * 0.2); // Visible range
                }
                
                // Update the new sky system
                this._updateSkySystem(phaseValue);
                
                // Update fog color and density based on day/night phase
                if (this._scene.fog) {
                    if (phaseValue <= 0.5) {
                        // Day fog - light blue-grey matching the sky
                        const dayFogColor = new THREE.Color(0x87CEEB); // Light blue-grey
                        this._scene.fog.color.copy(dayFogColor);
                        this._scene.fog.density = 0.015; // Very subtle
                    } else {
                        // Night fog - black
                        const nightFogColor = new THREE.Color(0x000000); // Black
                        this._scene.fog.color.copy(nightFogColor);
                        this._scene.fog.density = 0.005; // Same subtle density
                    }
                }
                
                if (this._rimLight) {
                    const nightColor = new THREE.Color(0x664422); // Warm torchlight glow
                    const dayColor = new THREE.Color(0xaa6633); // Warm orange
                    this._rimLight.color.copy(dayColor).lerp(nightColor, phaseValue);
                    this._rimLight.intensity = 0.3 - phaseValue * 0.1; // Visible rim light
                }
                
                if (this._hemiLight) {
                    const nightSkyColor = new THREE.Color(0x2a2a4a); // Dark blue night sky
                    const daySkyColor = new THREE.Color(0x6a7a9a); // Blue day sky
                    const nightGroundColor = new THREE.Color(0x2a1a0a); // Dark brown ground
                    const dayGroundColor = new THREE.Color(0x3a2a1a); // Brown ground
                    
                    this._hemiLight.color.copy(daySkyColor).lerp(nightSkyColor, phaseValue);
                    this._hemiLight.groundColor.copy(dayGroundColor).lerp(nightGroundColor, phaseValue);
                    this._hemiLight.intensity = 0.4 - phaseValue * 0.1; // Visible hemisphere light
                }
                
                // Update ambient light
                const ambientLight = this._scene.getObjectByName('ambientLight');
                if (ambientLight) {
                    const nightColor = new THREE.Color(0x3a3a5a); // Warm night ambient
                    const dayColor = new THREE.Color(0x4a4a3a); // Warm day ambient
                    ambientLight.color.copy(dayColor).lerp(nightColor, phaseValue);
                    ambientLight.intensity = 0.1; // + phaseValue * 0.25; // Increased night ambient from 0.15 to 0.25 for better visibility
                }
                // Fog removed - no fog transitions
                
                
                
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
                
                // Update bloom intensity based on phase - using user's preferred settings as base
                if (this._bloomPass) {
                    // Use user's daytime preferences as base, with slight adjustments for night
                    this._bloomPass.strength = 0.1 + phaseValue * 0.03; // User base: 0.07, slight increase at night (0.07 to 0.10)
                    this._bloomPass.radius = 0.08 + phaseValue * 0.04; // User base: 0.08, slight increase at night (0.08 to 0.12)
                    this._bloomPass.threshold = 0.00 + phaseValue * 0.1; // User base: 0.00, slight increase at night (0.00 to 0.10)
                }
              }

              _updateDynamicUI() {
                  if (!this._playerObjects || !this._camera) return;

                  const targetPos3D = new this._THREE.Vector3();
                  const bbox = new this._THREE.Box3();

                  this._playerObjects.forEach((player) => {
                      return;
                      const uiElement = player.playerUI?.element;
                      if (!uiElement) return;

                      const chatMessageCard = uiElement.querySelector('.chat-message-card');
                      const arrowEl = uiElement.querySelector('.bubble-arrow');

                      // Only run if the chat is active and we have the necessary elements
                      if (!chatMessageCard || !arrowEl || !player.model || !uiElement.classList.contains('chat-active')) {
                          if (arrowEl) arrowEl.style.visibility = 'hidden';
                          return;
                      }

                      // 1. Get the 3D model's center position
                      bbox.setFromObject(player.model, true);
                      bbox.getCenter(targetPos3D);

                      // 2. Project its 3D position to 2D screen coordinates
                      targetPos3D.project(this._camera);

                      // Hide if the character is behind the camera
                      if (targetPos3D.z > 1) {
                          arrowEl.style.visibility = 'hidden';
                          return;
                      }

                      // 3. Convert normalized screen coordinates to pixels
                      const targetX = (targetPos3D.x * 0.5 + 0.5) * this._width;
                      const targetY = (-targetPos3D.y * 0.5 + 0.5) * this._height;

                      // 4. Get the screen position of the chat bubble
                      const chatRect = chatMessageCard.getBoundingClientRect();
                      const parentRect = this._parent.getBoundingClientRect();
                      
                      // 5. The arrow's origin is always the middle of the chat bubble's left edge
                      const arrowOriginX = chatRect.left - parentRect.left;
                      const arrowOriginY = chatRect.top - parentRect.top + (chatRect.height / 2);

                      // 6. Calculate the angle from the arrow's origin to the character's screen position
                      const angle = Math.atan2(targetY - arrowOriginY, targetX - arrowOriginX);
                      const angleDeg = angle * (180 / Math.PI);

                      // 7. Make the arrow visible and apply the rotation
                      arrowEl.style.visibility = 'visible';
                      arrowEl.style.transform = `translateY(-50%) rotate(${angleDeg}deg)`;
                  });
              }

              _createPlayerUI(name, displayName, imageUrl, CSS2DObject) {
                  // Main container for the entire UI component (holds nameplate and chat card)
                  const container = document.createElement('div');
                  container.className = 'player-ui-container';

                  // --- Part 1: The Floating Player Info Card (Name, ID, Role, Timestamp, Avatar) ---
                  // This will *not* have a background/border to make it appear floating
                  const playerInfoCard = document.createElement('div');
                  playerInfoCard.className = 'player-info-card centered-component';

                  // Avatar
                  const img = document.createElement('img');
                  img.className = 'player-avatar-3d';
                  img.src = imageUrl;
                  playerInfoCard.appendChild(img);

                  // Text container for name, ID, role
                  const textDetails = document.createElement('div');
                  textDetails.className = 'player-text-details';

                  const nameText = document.createElement('div');
                  nameText.className = 'player-name-3d';
                  nameText.textContent = displayName || name;
                  textDetails.appendChild(nameText);

                  const playerIdText = document.createElement('div');
                  playerIdText.className = 'player-id-3d';
                  playerIdText.textContent = name; // Player ID / URL
                  textDetails.appendChild(playerIdText);

                  const roleText = document.createElement('div');
                  roleText.className = 'player-role-3d';
                  roleText.textContent = 'Role: Unknown'; // Updated dynamically
                  textDetails.appendChild(roleText);
                  
                  playerInfoCard.appendChild(textDetails);

                  // --- Part 2: The Hidden/Expanding Chat Message Card ---
                  const chatMessageCard = document.createElement('div');
                  chatMessageCard.className = 'chat-message-card'; // This remains a solid card
                  chatMessageCard.innerHTML = `
                      <div class="bubble-message"></div>
                  `;

                  // Assemble the component: Info card on left, chat card on right (initially hidden)
                  // container.appendChild(playerInfoCard);
                  // container.appendChild(chatMessageCard);
                  playerInfoCard.appendChild(chatMessageCard);
                  container.appendChild(playerInfoCard);

                  // Make the info card clickable to focus the camera
                  playerInfoCard.onclick = () => {
                      if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
                          const leftPanel = document.querySelector('.left-panel');
                          const rightPanel = document.querySelector('.right-panel');
                          const leftPanelWidth = leftPanel ? leftPanel.offsetWidth : 0;
                          const rightPanelWidth = rightPanel ? rightPanel.offsetWidth : 0;
                          window.werewolfThreeJs.demo.focusOnPlayer(name, leftPanelWidth, rightPanelWidth);
                      }
                  };

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
                // Initialize animation clock if not exists
                if (!this._animationClock) {
                    this._animationClock = new this._THREE.Clock();
                }
                
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
                  
                  // Animate clouds
                  if (this._clouds) {
                    this._clouds.forEach(cloud => {
                      if (!cloud || !cloud.position || !cloud.userData) return;
                      const userData = cloud.userData;
                      // Slowly rotate clouds around the scene
                      userData.initialAngle += userData.speed;
                      cloud.position.x = Math.cos(userData.initialAngle) * userData.radius;
                      cloud.position.z = Math.sin(userData.initialAngle) * userData.radius;
                      
                      // Gentle vertical bobbing
                      cloud.position.y = userData.height + Math.sin(time * 0.0005) * 2;
                    });
                  }
                  
                  // Animate stars twinkling
                  if (this._stars && this._phaseTransition && this._phaseTransition.current > 0.5) {
                    const sizes = this._stars.geometry.attributes.size.array;
                    for (let i = 0; i < sizes.length; i++) {
                      sizes[i] = (Math.random() * 2 + 0.5) * (0.8 + Math.sin(time * 0.001 + i) * 0.2);
                    }
                    this._stars.geometry.attributes.size.needsUpdate = true;
                  }
                  
                  // Animate god rays
                  if (this._godRayGroup && this._godRayGroup.visible && this._godRays) {
                    this._godRays.forEach((ray, index) => {
                      if (ray.userData) {
                        // Very subtle pulsing effect for ethereal appearance
                        const pulse = Math.sin(time * 0.0008 * ray.userData.speed + ray.userData.phase);
                        if (ray.material) {
                          // Animate opacity with gentle pulsing
                          const baseOpacity = ray.userData.originalOpacity * this._godRayIntensity;
                          ray.material.opacity = baseOpacity * (0.85 + pulse * 0.15); // Subtle variation
                        }
                        
                        // Gentle rotation to simulate atmospheric movement
                        ray.rotation.z = ray.userData.baseRotationZ + Math.sin(time * 0.0002 + index * 0.5) * 0.03;
                        
                        // Very subtle tilt variation for organic movement
                        ray.rotation.x = ray.userData.baseRotationX + Math.cos(time * 0.00025 + index * 0.3) * 0.02;
                        
                        // Slight length variation for breathing effect
                        const lengthVariation = 1 + Math.sin(time * 0.0003 + ray.userData.phase) * 0.05;
                        ray.scale.y = lengthVariation;
                      }
                    });
                  }
                  
                  // Animate moon rotation
                  if (this._moonMesh && this._moonMesh.visible) {
                    this._moonMesh.rotation.y = time * 0.00002;
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
                    // Get proper delta time from clock
                    const delta = this._animationClock ? this._animationClock.getDelta() : 0.016;
                    
                    this._playerObjects.forEach((player, name) => {
                      // Update animation mixers with proper delta
                      if (player.mixer) {
                        player.mixer.update(delta);
                      }
                      
                      if (player.isAlive) {
                        // Enhanced floating animation for alive players
                        const floatOffset = Math.sin(time * 0.001 + player.baseAngle) * 0.2;
                        const bobOffset = Math.cos(time * 0.0015 + player.baseAngle * 2) * 0.05;
                        player.container.position.y = floatOffset + bobOffset;
                        
                        // More dynamic orb rotation
                        if (player.orb) {
                          player.orb.rotation.y = time * 0.003;
                          player.orb.rotation.x = Math.sin(time * 0.002) * 0.15;
                          player.orb.rotation.z = Math.cos(time * 0.0025) * 0.1;
                        }
                        
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

                 // Debug: Log sky visibility on first few frames
                 if (!this._skyDebugLogged || this._skyDebugFrameCount < 5) {
                   if (!this._skyDebugFrameCount) this._skyDebugFrameCount = 0;
                   this._skyDebugFrameCount++;
                   
                   if (this._sky) {
                    //  console.debug(`[SKY DEBUG] Frame ${this._skyDebugFrameCount} - Sky mesh status:`, {
                    //    visible: this._sky.visible,
                    //    inScene: this._scene.children.includes(this._sky),
                    //    scale: this._sky.scale.x,
                    //    position: {x: this._sky.position.x, y: this._sky.position.y, z: this._sky.position.z},
                    //    renderOrder: this._sky.renderOrder,
                    //    material: this._sky.material ? 'exists' : 'missing',
                    //    uniforms: this._sky.material ? {
                    //      sunPosition: this._sky.material.uniforms['sunPosition'].value,
                    //      turbidity: this._sky.material.uniforms['turbidity'].value,
                    //      rayleigh: this._sky.material.uniforms['rayleigh'].value
                    //    } : 'N/A'
                    //  });
                   } else {
                     console.error('[SKY DEBUG] Sky mesh is null/undefined!');
                   }
                   
                   if (this._skyDebugFrameCount >= 5) {
                     this._skyDebugLogged = true;
                   }
                 }

                 this._updateDynamicUI();

                 // Use post-processing composer if available, otherwise fallback to direct render
                 if (this._composer) {
                   this._composer.render();
                 } else {
                   this._threejs.render(this._scene, this._camera);
                 }
                 this._labelRenderer.render(this._scene, this._camera);
                 
                 // Update campfire
                 if (this._fire && this._fire.update) {
                   const elapsed = time * 0.001;
                   this._fire.update(elapsed);
                 }
                 
                 // Animate fire light
                 if (this._fireLight) {
                   this._fireLight.intensity = 4.5 + Math.sin(time * 0.003) * 0.5;
                 }
                 
                 this._RAF();
                });
              }
              
              // Method to update sky for phase (exposed for test page)
              updateSkyForPhase(isDay) {
                const targetPhase = isDay ? 0.25 : 0.75; // Mid-day or mid-night
                this._updateSkySystem(targetPhase);
                this._updateSceneForPhase(targetPhase);
              }
            }

            setupScene(BasicWorldDemo, VolumetricFire);
        } catch (error) {
            console.error("Failed to load Three.js modules:", error);
            parent.textContent = "Error loading 3D assets. Please refresh.";
        }
    };

    loadAndSetup();
  }

  function setupScene(BasicWorldDemo, VolumetricFire) {
    if (threeState.initialized) return;
    threeState.demo = new BasicWorldDemo({ parent, width, height });
    threeState.initialized = true;
    
    // Expose the demo instance globally for the test page
    window.werewolfThreeJs.demo = threeState.demo;
  }

  function updateSceneFromGameState(gameState, playerMap, actingPlayerName) {
    if (!threeState.demo || !threeState.demo._playerObjects) return;

    // --- Hide all chat bubbles at the start of each step ---
    threeState.demo._playerObjects.forEach(player => {
        if (player.playerUI && player.playerUI.element) {
            player.playerUI.element.classList.remove('chat-active');
        }
    });

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

      threeState.demo.updatePlayerStatus(player.name, player, primaryStatus, threatLevel);
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
        let messageForBubble = '';
        let reasoningForBubble = lastEvent.reasoning || '';
        const actorName = lastEvent.actor_id || lastEvent.speaker;
        
        // Determine the message for the bubble based on event type
        switch(lastEvent.type) {
            case 'chat':
                messageForBubble = `"${lastEvent.message}"`;
                if(threeState.demo.triggerSpeakingAnimation) threeState.demo.triggerSpeakingAnimation(actorName);
                break;
            case 'vote':
            case 'night_vote':
                messageForBubble = `Votes for <strong>${lastEvent.target}</strong>.`;
                if(threeState.demo.triggerPointingAnimation) threeState.demo.triggerPointingAnimation(actorName);
                break;
            case 'doctor_heal_action':
                messageForBubble = `Heals <strong>${lastEvent.target}</strong>.`;
                break;
            case 'seer_inspection':
                messageForBubble = `Inspects <strong>${lastEvent.target}</strong>.`;
                break;
        }

        // Display the bubble if we have a message and an actor
        if (messageForBubble && actorName && playerMap.has(actorName)) {
            const formattedMessage = window.werewolfGamePlayer.playerIdReplacer(messageForBubble);
            threeState.demo.displayPlayerBubble(actorName, formattedMessage, reasoningForBubble, lastEvent.timestamp); 
            threeState.demo.updatePlayerActive(actorName); // Keep the active pulse effect
        }

        // Handle game_over event animations (these don't have bubbles)
        if (lastEvent.type === 'game_over') {
            if (lastEvent.winners && threeState.demo.triggerVictoryAnimation) {
                lastEvent.winners.forEach(winnerName => {
                    if (playerMap.has(winnerName)) {
                        threeState.demo.triggerVictoryAnimation(winnerName);
                    }
                });
            }
            if (lastEvent.losers && threeState.demo.triggerDefeatedAnimation) {
                lastEvent.losers.forEach(loserName => {
                    if (playerMap.has(loserName)) {
                        threeState.demo.triggerDefeatedAnimation(loserName);
                    }
                });
            }
        }

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
            
            // Trigger pointing animation for vote-related events
            if ((lastEvent.type === 'vote' || lastEvent.type === 'night_vote') && threeState.demo.triggerPointingAnimation) {
                threeState.demo.triggerPointingAnimation(actorName);
            }
        }
        
        // Handle game_over event animations
        if (lastEvent.type === 'game_over') {
            // Trigger victory animation for winners
            if (lastEvent.winners && threeState.demo.triggerVictoryAnimation) {
                lastEvent.winners.forEach(winnerName => {
                    if (playerMap.has(winnerName)) {
                        threeState.demo.triggerVictoryAnimation(winnerName);
                    }
                });
            }
            
            // Trigger defeated animation for losers
            if (lastEvent.losers && threeState.demo.triggerDefeatedAnimation) {
                lastEvent.losers.forEach(loserName => {
                    if (playerMap.has(loserName)) {
                        threeState.demo.triggerDefeatedAnimation(loserName);
                    }
                });
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
        
        .game-scoreboard .phase-indicator {
            position: relative; /* Override the 'fixed' positioning */
            top: auto;
            left: auto;
            transform: none;
            z-index: auto;
            scale: 1.0;
            
            /* Scale it down to fit */
            padding: 6px 14px;
            font-size: 0.9rem;
            border-radius: 20px;
            
            /* Remove styles meant for a floating element */
            box-shadow: none;
            backdrop-filter: none;
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
            /* left: 20px; */
            /* width: 320px; */
            display: none; /* Hide the left panel */
        }

        .right-panel {
            position: fixed;
            top: 54px;
            left: 20px;
            width: 420px;
            max-height: calc(100vh - 124px);
            background: var(--panel-bg);
            backdrop-filter: blur(20px) saturate(1.5);
            border-radius: 16px;
            border: 1px solid var(--panel-border);
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            pointer-events: auto;
            box-shadow: var(--card-shadow), 0 0 40px rgba(116, 185, 255, 0.05);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
        }

        .right-panel.collapsed {
            max-height: 65px; /* Height of just the header */
            padding-top: 0;
            padding-bottom: 0;
        }
        
        /* ENHANCED Header to act as a toggle */
        .right-panel h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text-primary);
            position: relative;
            padding: 20px 20px 20px 60px; 
            flex-shrink: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            user-select: none;
        }

        /* Adds a visual indicator for expanding/collapsing */
        .right-panel h1::before {
            content: ''; /* The content is now the background image */
            position: absolute;
            left: 20px;
            width: 20px;
            height: 20px;

            background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%2374b9ff"><path d="M12 8l-6 6h12z"/></svg>');
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;

            transition: transform 0.3s ease;
        }

        .right-panel.collapsed h1::before {
            transform: rotate(180deg);
        }

        .right-panel h1 > span {
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Hides the log when panel is collapsed */
        .right-panel.collapsed #chat-log {
            display: none;
        }

        /* Main container: now acts as a positioning context for its children.
          Its OWN top-left corner is anchored by the CSS2DObject's 3D position.
          We will center its *content* using flex or absolute positioning.
        */
        .player-ui-container {
            position: relative; /* CRITICAL: Establishes a positioning context for absolute children */
            display: block; /* No longer flex. Children handle their own layout */
            width: max-content; /* Allow container to grow with content */
            height: max-content; /* Allow container to grow with content */
            pointer-events: none; /* Let children handle clicks */

            /* CRITICAL ANCHORING FIX:
              We want the player-info-card (nameplate) to be horizontally centered
              on the 3D point. So, we'll position the info card *absolutely*
              within this container, with its own transform to center itself.
              The container itself is placed at 0,0 relative to the CSS2DObject.
            */
            transform: none; /* REMOVE any transforms from the container itself */
        }

        /* The truly FLOATING player info card (Name, ID, Role, Timestamp, Avatar)
          This is now absolutely positioned and self-centered within the container.
        */
        .player-info-card {
            position: relative;
            top: 50%; /* Position its top edge at the vertical center of the container */
            left: 50%; /* Position its left edge at the horizontal center of the container */
            
            /* CRITICAL: Translate it back by 50% of its OWN width and height to truly center it.
              This element is now fixed relative to the 3D point.
            */
            transform: translate(-50%, -50%);
            
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
            
            /* TRULY FLOATING AESTHETIC */
            background: transparent;
            backdrop-filter: none;
            border: none;
            box-shadow: none;
            
            color: #ffffff;
            text-shadow: 
                0 0 4px rgba(0,0,0,0.8),
                0 0 8px rgba(0,0,0,0.6);
            pointer-events: auto;
            cursor: pointer;
            transition: all 0.2s ease;
            padding: 0;
            margin: 0;
            white-space: nowrap; /* Prevent text wrapping */
            z-index: 5;

            padding: 4px 8px;
            border: 1px solid transparent;
            border-radius: 8px;
            // transition: border-color 0.2s ease
        }

        .player-info-card:hover {
            color: #ffcc00;
            text-shadow: 
                0 0 6px rgba(0,0,0,0.9), 
                0 0 10px rgba(0,0,0,0.7),
                0 0 12px rgba(255,204,0,0.4);
        }

        /* NEW: This encircles the nameplate when chat is active */
        .player-ui-container.chat-active .player-info-card {
            // background: rgba(10, 10, 20, 0.4);
            // background: transparent;
            border: 1px solid #ffffff;
            box-shadow: 0 0 2px rgba(255, 255, 255, 0.8), 0 0 12px rgba(255, 255, 255, 0.5);
            border-radius: 8px;
            padding: 4px 8px; /* Add padding so border isn't flush */
        }

        /* NEW: This styles the message text like a movie subtitle */
        .player-ui-container.chat-active .bubble-message {
            font-size: 14px;
            font-weight: 500;
            color: #ffffff;
            line-height: 1.4;
            /* Classic subtitle shadow */
            text-shadow: 
                0 0 5px rgba(0,0,0,1), 
                0 0 8px rgba(0,0,0,0.8);
        }

        .player-avatar-3d {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            object-fit: cover;
            background-color: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.4);
            box-shadow: 0 0 5px rgba(0,0,0,0.5);
            margin-bottom: 4px;
        }

        .player-text-details {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .player-name-3d {
            font-size: 14px;
            font-weight: 700;
        }
        .player-id-3d {
            font-size: 10px;
            opacity: 0.9;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        }
        .player-role-3d {
            font-size: 11px;
            opacity: 0.8;
        }

        /* The chat message card (initially hidden, expands to the right *of the player-info-card*) */
        .chat-message-card {
            position: absolute;
            
            /* Position the bubble's top edge at the nameplate's vertical center */
            top: 50%;
            
            /* Position the bubble's LEFT edge at the nameplate's RIGHT edge */
            left: 100%;
            
            /* Adjust for perfect vertical centering and add a gap */
            transform: translateY(-50%);
            margin-left: 12px;
            
            /* Make sure the bubble is on top */
            z-index: 10;

            min-width: 180px;
            
            background: transparent;
            border: none;
            box-shadow: none;
            max-width: 250px;

            /* Initial Hidden State */
            opacity: 0;
            transition: opacity 0.1s ease;
            white-space: normal;

            transition-property: opacity;
            transition-duration: 0.2s;
            z-index: 10;
        }

        /* Active state: chat message card expands */
        .player-ui-container.chat-active .chat-message-card {
            opacity: 1;
        }

        .chat-message-card .bubble-message {
            font-size: 14px;
            line-height: 1.4;
        }
        
        /* --- End New Unified Player UI Component --- */
        
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
            opacity: 0.85; /* Increased from 0.5 for better visibility */
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
        /* Sky Controls Panel */
        .sky-controls-panel {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: linear-gradient(to bottom, rgba(26, 26, 46, 0.95), rgba(26, 26, 46, 1));
            backdrop-filter: blur(15px);
            border: 1px solid rgba(116, 185, 255, 0.3);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            pointer-events: auto;
            max-height: 25vh;
            overflow-y: auto;
            transition: all 0.3s ease;
        }
        
        .sky-controls-panel.collapsed {
            max-height: 50px;
            overflow: hidden;
        }
        
        .sky-controls-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            cursor: pointer;
            user-select: none;
        }
        
        .sky-controls-title {
            font-size: 14px;
            font-weight: bold;
            color: #667eea;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .sky-controls-toggle {
            font-size: 18px;
            color: #667eea;
            transition: transform 0.3s ease;
        }
        
        .sky-controls-panel.collapsed .sky-controls-toggle {
            transform: rotate(180deg);
        }
        
        .sky-controls-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }
        
        .sky-control-section {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 12px;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        .sky-section-title {
            font-size: 12px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid rgba(102, 126, 234, 0.3);
            padding-bottom: 5px;
        }
        
        .sky-control-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }
        
        .sky-control-label {
            font-size: 10px;
            color: #aaa;
            min-width: 120px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .sky-control-value {
            font-size: 10px;
            color: #667eea;
            font-weight: bold;
            min-width: 45px;
            text-align: right;
        }
        
        .sky-button-group {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        
        .sky-button {
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .sky-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .sky-button.day {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .sky-button.night {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        .sky-controls-panel input[type="range"] {
            flex: 1;
            height: 4px;
            background: #333;
            outline: none;
            border-radius: 2px;
            min-width: 80px;
        }
        
        .sky-controls-panel input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 14px;
            height: 14px;
            background: #667eea;
            cursor: pointer;
            border-radius: 50%;
            transition: all 0.2s ease;
        }
        
        .sky-controls-panel input[type="range"]::-webkit-slider-thumb:hover {
            background: #7c8ff0;
            transform: scale(1.2);
        }
        
        .sky-controls-panel input[type="range"]::-moz-range-thumb {
            width: 14px;
            height: 14px;
            background: #667eea;
            cursor: pointer;
            border-radius: 50%;
            border: none;
            transition: all 0.2s ease;
        }
        
        .sky-controls-panel input[type="range"]::-moz-range-thumb:hover {
            background: #7c8ff0;
            transform: scale(1.2);
        }
        
        .sky-controls-panel input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }
        
        .sky-info-panel {
            background: rgba(0, 0, 0, 0.3);
            padding: 8px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 10px;
            line-height: 1.4;
        }
        
        .sky-info-item {
            display: flex;
            justify-content: space-between;
            padding: 2px 0;
        }
        
        .sky-info-label {
            color: #888;
        }
        
        .sky-info-value {
            color: #fff;
            font-weight: bold;
        }
        
        .sky-phase-indicator {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 10px;
        }
        
        .sky-phase-day {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        }
        
        .sky-phase-night {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
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
    
    // Update phase indicator based on current game state
    const currentPhase = allEvents[eventStep].phase.toUpperCase() || 'DAY';
    const isNight = currentPhase === 'NIGHT';
    
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
            <div id="phase-indicator-capsule" class="phase-indicator">
            </div>
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

    const phaseCapsule = scoreboard.querySelector('#phase-indicator-capsule');
    if (phaseCapsule) {
        const currentPhase = (allEvents[eventStep]?.phase || 'DAY').toUpperCase();
        const isNight = currentPhase === 'NIGHT';
        
        // Set the class for 'day' or 'night' styling
        phaseCapsule.className = `phase-indicator ${isNight ? 'night' : 'day'}`;
        
        // Set the inner content with the icon and phase name
        const phaseIcon = isNight ? '&#x1F319;' : '&#x2600;';
        if (allEvents[eventStep]?.event_name == 'game_end') {
            phaseCapsule.innerHTML = `
            <span class="phase-icon">${isNight ? '&#x1F319;' : '&#x2600;'}</span>
            `;
        } else {
            phaseCapsule.innerHTML = `
            <span class="phase-icon">${isNight ? '&#x1F319;' : '&#x2600;'}</span>
            <span>${allEvents[eventStep].day}</span>
            `;
        }
    }

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
    updateEventLog(rightPanel, gameState, playerMap);

    // Create sky controls panel if it doesn't exist
    // createSkyControlsPanel(parent);

    // Update 3D scene based on game state
    updateSceneFromGameState(gameState, playerMap, nameToHighlight);
    
    // Initialize 3D players if needed - check the flag to prevent duplicate initialization
    if (threeState.demo && threeState.demo._playerObjects && !threeState.players3DInitialized && playerNamesFor3D.length > 0) {
        console.debug('Starting 3D player initialization...');
        threeState.players3DInitialized = true;  // Set flag immediately to prevent duplicate calls
        initializePlayers3D(gameState, playerNamesFor3D, playerThumbnailsFor3D, threeState).then(() => {
            console.debug('3D players initialized with FBX models');
            // Update scene after models are loaded
            updateSceneFromGameState(gameState, playerMap, nameToHighlight);
        }).catch(error => {
            console.error('Failed to initialize 3D players:', error);
            // Reset flag on error to allow retry
            threeState.players3DInitialized = false;
        });
    }
}

async function initializePlayers3D(gameState, playerNames, playerThumbnails, threeState) {
    if (!threeState || !threeState.demo || !threeState.demo._playerObjects) return;
    
    console.debug(`initializePlayers3D called with ${playerNames.length} players`);
    
    // Double-check the flag to ensure we're not already initialized
    if (threeState.players3DInitialized && threeState.demo._playerObjects.size > 0) {
        console.warn('3D players already initialized, skipping duplicate initialization');
        return;
    }
    
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
    
    // Skip creating platform geometry since we're using the island model
    // The island model loaded in _LoadModels provides the ground
    
    // Add decorative lines from center to each player position
    const linesMaterial = new THREE.LineBasicMaterial({
        color: 0x334455,
        transparent: true,
        opacity: 0.3
    });
    
    // Load all models (which now include animations) concurrently
    const playerLoadPromises = playerNames.map(async (name, i) => {
        const role = gameState.players[i].role || 'Villager';
        try {
            // Load the merged FBX model (animations are extracted automatically)
            const fbxModel = await threeState.demo.loadCharacterModel(role);
            // Get the animations that were extracted during model loading
            const animations = await threeState.demo.loadCharacterAnimations(role);
            return { name, i, role, fbxModel, animations, success: true };
        } catch (error) {
            console.error(`Failed to load merged model for ${name}:`, error);
            return { name, i, role, fbxModel: null, animations: null, success: false };
        }
    });
    
    const loadedPlayers = await Promise.all(playerLoadPromises);
    
    // Create player objects with loaded models
    loadedPlayers.forEach(({ name, i, role, fbxModel, animations, success }) => {
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
            color: 0x1a1a2a, // Darker base color
            roughness: 0.9, // More matte
            metalness: 0.1, // Less metallic
            emissive: 0x0a0a15, // Very subtle glow
            emissiveIntensity: 0.1 // Minimal emission
        });
        const pedestal = new THREE.Mesh(pedestalGeometry, pedestalMaterial);
        pedestal.position.y = 0.2;
        pedestal.castShadow = true;
        pedestal.receiveShadow = true;
        playerContainer.add(pedestal);
        
        let model = null;
        let mixer = null;
        let currentAction = null;
        let modelHeight = playerHeight;
        
        if (success && fbxModel) {
            // Clone the FBX model to ensure each player gets their own instance
            model = threeState.demo._skeletonUtils.clone(fbxModel);
            model.position.y = 0.5; // Slightly higher to account for larger model
            
            // Enable shadows for all meshes in the model
            model.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    
                    // Modify existing materials to be matte (don't replace to preserve skinning)
                    if (child.material) {
                        // Handle both single material and array of materials
                        const materials = Array.isArray(child.material) ? child.material : [child.material];
                        materials.forEach(mat => {
                            if (mat) {
                                // Modify the existing material to be completely matte
                                mat.roughness = 1.0;  // Maximum roughness = no shine
                                mat.metalness = 0.0;  // No metallic properties
                                mat.envMapIntensity = 0;  // No environment reflections
                                // mat.emissiveIntensity = 0.5
                                
                                // Remove maps that might add shine
                                mat.roughnessMap = null;
                                mat.metalnessMap = null;
                                mat.envMap = null;
                                
                                // Brighten the color to compensate for matte surface
                                if (mat.color) {
                                    mat.color.multiplyScalar(1.5);
                                }
                                
                                // Ensure proper settings
                                mat.transparent = false;
                                mat.opacity = 1.0;
                                mat.alphaTest = 0;
                                mat.needsUpdate = true;
                            }
                        });
                    }
                }
            });
            
            playerContainer.add(model);
            
            // Create AnimationMixer
            mixer = new THREE.AnimationMixer(model);
            
            // Play Idle animation if available
            if (animations && animations['Idle']) {
                currentAction = mixer.clipAction(animations['Idle']);
                currentAction.play();
            }
            
            // Calculate model height using bounding box
            const box = new THREE.Box3().setFromObject(model);
            const size = box.getSize(new THREE.Vector3());
            modelHeight = size.y;
        } else {
            // Create geometric fallback (colored cube)
            console.warn(`Using geometric fallback for ${name} (role: ${role})`);
            
            const fallbackGeometry = new THREE.BoxGeometry(1.5, 3, 1.5);
            const fallbackColor = role === 'Werewolf' ? 0x880000 :
                                 role === 'Doctor' ? 0x008800 :
                                 role === 'Seer' ? 0x4B0082 : 0x4466ff;
            const fallbackMaterial = new THREE.MeshStandardMaterial({
                color: fallbackColor,
                roughness: 0.5,
                metalness: 0.3,
                emissive: fallbackColor,
                emissiveIntensity: 0.2
            });
            const fallback = new THREE.Mesh(fallbackGeometry, fallbackMaterial);
            fallback.position.y = 2;
            fallback.castShadow = true;
            fallback.receiveShadow = true;
            playerContainer.add(fallback);
            
            model = fallback;
            modelHeight = 3;
        }
        
        // Create glowing orb for status (positioned above character's head)
        const orbGeometry = new THREE.IcosahedronGeometry(0.25, 2); // Smaller orb
        const orbMaterial = new THREE.MeshStandardMaterial({
            color: 0x00aa88, // More muted green-blue
            emissive: 0x00aa88,
            emissiveIntensity: 1.,
            transparent: true,
            opacity: 1., // More translucent
            depthTest: false
        });
        const orb = new THREE.Mesh(orbGeometry, orbMaterial);
        orb.position.y = modelHeight + 0.8; // Position above model
        orb.name = 'statusOrb';
        playerContainer.add(orb);
        
        // Add outer glow sphere - more subtle
        const glowGeometry = new THREE.SphereGeometry(0.4, 12, 8); // Smaller glow
        const glowMaterial = new THREE.MeshStandardMaterial({
            color: 0x00aa88, // Muted color
            emissive: 0x00aa88,
            emissiveIntensity: 0.15, // Very subtle glow
            transparent: true,
            opacity: 0.2, // More transparent
            depthTest: false
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.y = modelHeight + 0.8; // Position above model
        playerContainer.add(glow);
        
        // Add point light for glow effect - dimmer
        const orbLight = new THREE.PointLight(0x00aa88, 0.4, 6); // Less intense, smaller radius
        orbLight.position.y = modelHeight + 0.8; // Position above model
        orbLight.name = 'orbLight';
        orbLight.castShadow = true;
        playerContainer.add(orbLight);
        
        // Make player face center without flipping
        // Calculate the angle to face the center
        // Update rotation to use negative values so players face toward the origin
        const angleToCenter = Math.atan2(
            playerContainer.position.x,
            playerContainer.position.z
        );

        // Apply the rotation. The "+ Math.PI" is a 180-degree turn needed to face the target.
        // If your models still face the wrong way, try one of the options from the troubleshooting section below.
        playerContainer.rotation.y = angleToCenter + Math.PI;
        
        // Create nameplate with actual player thumbnail (positioned above orb)
        const thumbnailUrl = playerThumbnails[name] || `https://via.placeholder.com/60/2c3e50/ecf0f1?text=${name.charAt(0)}`;

        const playerUI = threeState.demo._createPlayerUI(name, displayName, thumbnailUrl, CSS2DObject);
        playerUI.position.set(0, modelHeight + 2.0, 0); // Position it above the model
        playerContainer.add(playerUI);
        
        // Store references with new structure
        threeState.demo._playerObjects.set(name, {
            container: playerContainer,
            model: model,
            playerUI: playerUI,
            mixer: mixer,
            animations: animations,
            currentAction: currentAction,
            orb: orb,
            glow: glow,
            orbLight: orbLight,
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




// --- Sky Controls Functions ---
function createSkyControlsPanel(parent) {
    // Check if panel already exists
    let panel = parent.querySelector('.sky-controls-panel');
    if (panel) return panel;
    
    panel = document.createElement('div');
    panel.className = 'sky-controls-panel collapsed'; 
    panel.innerHTML = `
        <div class="sky-controls-header" onclick="toggleSkyControls()">
            <div class="sky-controls-title">🌤️ Sky & Lighting Controls</div>
            <div class="sky-controls-toggle">▼</div>
        </div>
        <div class="sky-controls-content">
            <div class="sky-button-group">
                <button class="sky-button day" onclick="setSkyDayTime()">☀️ Day</button>
                <button class="sky-button night" onclick="setSkyNightTime()">🌙 Night</button>
                <button class="sky-button" onclick="toggleSkyTransition()">🔄 Auto</button>
            </div>
            
            <!-- Sky Parameters -->
            <div class="sky-control-section">
                <div class="sky-section-title">Sky Parameters</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Turbidity</span>
                        <span class="sky-control-value" id="sky-turbidity-value">10.0</span>
                    </div>
                    <input type="range" id="sky-turbidity" min="0" max="20" step="0.1" value="10" oninput="updateSkyParameter('turbidity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Rayleigh</span>
                        <span class="sky-control-value" id="sky-rayleigh-value">2.0</span>
                    </div>
                    <input type="range" id="sky-rayleigh" min="0" max="10" step="0.1" value="2" oninput="updateSkyParameter('rayleigh', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Mie Coeff</span>
                        <span class="sky-control-value" id="sky-mie-coeff-value">0.005</span>
                    </div>
                    <input type="range" id="sky-mie-coeff" min="0" max="0.1" step="0.001" value="0.005" oninput="updateSkyParameter('mieCoefficient', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Mie Dir G</span>
                        <span class="sky-control-value" id="sky-mie-g-value">0.8</span>
                    </div>
                    <input type="range" id="sky-mie-g" min="0" max="1" step="0.01" value="0.8" oninput="updateSkyParameter('mieDirectionalG', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Elevation</span>
                        <span class="sky-control-value" id="sky-elevation-value">45°</span>
                    </div>
                    <input type="range" id="sky-elevation" min="-90" max="90" step="1" value="45" oninput="updateSunPosition('elevation', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Azimuth</span>
                        <span class="sky-control-value" id="sky-azimuth-value">180°</span>
                    </div>
                    <input type="range" id="sky-azimuth" min="0" max="360" step="1" value="180" oninput="updateSunPosition('azimuth', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Exposure</span>
                        <span class="sky-control-value" id="sky-exposure-value">0.5</span>
                    </div>
                    <input type="range" id="sky-exposure" min="0" max="2" step="0.01" value="1.0" oninput="updateExposure(this.value)">
                </div>
            </div>
            
            <!-- Lighting -->
            <div class="sky-control-section">
                <div class="sky-section-title">Lighting</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Sun Intensity</span>
                        <span class="sky-control-value" id="sky-sun-intensity-value">1.0</span>
                    </div>
                    <input type="range" id="sky-sun-intensity" min="0" max="3" step="0.1" value="1" oninput="updateLighting('sunIntensity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Moon Intensity</span>
                        <span class="sky-control-value" id="sky-moon-intensity-value">0.5</span>
                    </div>
                    <input type="range" id="sky-moon-intensity" min="0" max="1" step="0.01" value="0.5" oninput="updateLighting('moonIntensity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Ambient</span>
                        <span class="sky-control-value" id="sky-ambient-value">0.4</span>
                    </div>
                    <input type="range" id="sky-ambient" min="0" max="1" step="0.01" value="0.1" oninput="updateLighting('ambientIntensity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Time of Day</span>
                        <span class="sky-control-value" id="sky-time-value">12:00</span>
                    </div>
                    <input type="range" id="sky-time-slider" min="0" max="24" step="0.1" value="12" oninput="setTimeOfDay(this.value)">
                </div>
            </div>
            
            <!-- Post-Processing -->
            <div class="sky-control-section">
                <div class="sky-section-title">Bloom</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Strength</span>
                        <span class="sky-control-value" id="sky-bloom-strength-value">0.6</span>
                    </div>
                    <input type="range" id="sky-bloom-strength" min="0" max="1" step="0.01" value="0.6" oninput="updateBloom('strength', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Radius</span>
                        <span class="sky-control-value" id="sky-bloom-radius-value">2.0</span>
                    </div>
                    <input type="range" id="sky-bloom-radius" min="0" max="2" step="0.01" value="2.0" oninput="updateBloom('radius', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Threshold</span>
                        <span class="sky-control-value" id="sky-bloom-threshold-value">0.1</span>
                    </div>
                    <input type="range" id="sky-bloom-threshold" min="0" max="1" step="0.01" value="0.1" oninput="updateBloom('threshold', this.value)">
                </div>
            </div>
            
            <!-- Clouds -->
            <div class="sky-control-section">
                <div class="sky-section-title">Clouds</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Opacity</span>
                        <span class="sky-control-value" id="sky-cloud-opacity-value">0.5</span>
                    </div>
                    <input type="range" id="sky-cloud-opacity" min="0" max="1" step="0.01" value="0.5" oninput="updateClouds('opacity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Speed</span>
                        <span class="sky-control-value" id="sky-cloud-speed-value">1.0</span>
                    </div>
                    <input type="range" id="sky-cloud-speed" min="0" max="5" step="0.1" value="1" oninput="updateClouds('speed', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Enable</span>
                    </div>
                    <input type="checkbox" id="sky-cloud-toggle" checked onchange="toggleClouds(this.checked)">
                </div>
            </div>
            
            <!-- God Rays -->
            <div class="sky-control-section">
                <div class="sky-section-title">God Rays</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Intensity</span>
                        <span class="sky-control-value" id="sky-godray-intensity-value">2.0</span>
                    </div>
                    <input type="range" id="sky-godray-intensity" min="0" max="2" step="0.1" value="2.0" oninput="updateGodRays('intensity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Enable</span>
                    </div>
                    <input type="checkbox" id="sky-godray-toggle" checked onchange="toggleGodRays(this.checked)">
                </div>
            </div>
            
            <!-- Status -->
            <div class="sky-control-section">
                <div class="sky-section-title">Status</div>
                <div class="sky-info-panel">
                    <div class="sky-info-item">
                        <span class="sky-info-label">Phase:</span>
                        <span class="sky-info-value"><span id="sky-current-phase" class="sky-phase-indicator sky-phase-day">DAY</span></span>
                    </div>
                    <div class="sky-info-item">
                        <span class="sky-info-label">Sun:</span>
                        <span class="sky-info-value" id="sky-sun-position">Az: 180°, El: 45°</span>
                    </div>
                    <div class="sky-info-item">
                        <span class="sky-info-label">Moon:</span>
                        <span class="sky-info-value" id="sky-moon-position">Hidden</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    parent.appendChild(panel);
    return panel;
}

function toggleSkyControls() {
    const panel = document.querySelector('.sky-controls-panel');
    if (panel) {
        panel.classList.toggle('collapsed');
    }
}

function updateSkyParameter(param, value) {
    const floatValue = parseFloat(value);
    
    // Update display
    let displayValue = floatValue.toFixed(param === 'mieCoefficient' ? 3 : 1);
    const valueId = param === 'mieCoefficient' ? 'sky-mie-coeff-value' : 
                    param === 'mieDirectionalG' ? 'sky-mie-g-value' : 
                    `sky-${param}-value`;
    const valueEl = document.getElementById(valueId);
    if (valueEl) valueEl.textContent = displayValue;
    
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._sky) {
        const sky = window.werewolfThreeJs.demo._sky;
        if (sky.material && sky.material.uniforms && sky.material.uniforms[param]) {
            sky.material.uniforms[param].value = floatValue;
            console.debug(`[Sky Controls] Updated ${param} to ${displayValue}`);
        }
    }
}

function updateSunPosition(type, value) {
    const floatValue = parseFloat(value);
    const valueEl = document.getElementById(`sky-${type}-value`);
    if (valueEl) valueEl.textContent = `${floatValue}°`;
    
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._sky) {
        const sky = window.werewolfThreeJs.demo._sky;
        
        // Get current values
        const elevationEl = document.getElementById('sky-elevation');
        const azimuthEl = document.getElementById('sky-azimuth');
        const currentElevation = elevationEl ? parseFloat(elevationEl.value) : 45;
        const currentAzimuth = azimuthEl ? parseFloat(azimuthEl.value) : 180;
        
        // Convert to radians
        const phi = (90 - currentElevation) * Math.PI / 180;
        const theta = currentAzimuth * Math.PI / 180;
        
        // Calculate sun position
        const sunX = Math.sin(phi) * Math.cos(theta);
        const sunY = Math.cos(phi);
        const sunZ = Math.sin(phi) * Math.sin(theta);
        
        if (sky.material && sky.material.uniforms && sky.material.uniforms['sunPosition']) {
            sky.material.uniforms['sunPosition'].value.set(sunX, sunY, sunZ);
        }
        
        // Update lighting
        updateDayNightLighting(currentElevation);
        updateSkyInfo();
        
        console.debug(`[Sky Controls] Sun position - Elevation: ${currentElevation}°, Azimuth: ${currentAzimuth}°`);
    }
}

function updateExposure(value) {
    const floatValue = parseFloat(value);
    const valueEl = document.getElementById('sky-exposure-value');
    if (valueEl) valueEl.textContent = floatValue.toFixed(2);
    
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._renderer) {
        window.werewolfThreeJs.demo._renderer.toneMappingExposure = floatValue;
        console.debug(`[Sky Controls] Updated exposure to ${floatValue.toFixed(2)}`);
    }
}

function updateLighting(type, value) {
    const floatValue = parseFloat(value);
    const valueId = type === 'sunIntensity' ? 'sky-sun-intensity-value' :
                    type === 'moonIntensity' ? 'sky-moon-intensity-value' :
                    'sky-ambient-value';
    const valueEl = document.getElementById(valueId);
    if (valueEl) valueEl.textContent = floatValue.toFixed(1);
    
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
        const demo = window.werewolfThreeJs.demo;
        
        switch(type) {
            case 'sunIntensity':
                if (demo._sunLight) {
                    demo._sunLight.intensity = floatValue;
                }
                break;
            case 'moonIntensity':
                if (demo._moonLight) {
                    demo._moonLight.intensity = floatValue;
                }
                break;
            case 'ambientIntensity':
                if (demo._ambientLight) {
                    demo._ambientLight.intensity = floatValue;
                }
                break;
        }
        
        console.debug(`[Sky Controls] Updated ${type} to ${floatValue.toFixed(1)}`);
    }
}

function updateBloom(type, value) {
    const floatValue = parseFloat(value);
    const valueId = `sky-bloom-${type}-value`;
    const valueEl = document.getElementById(valueId);
    if (valueEl) valueEl.textContent = floatValue.toFixed(2);
    
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._bloomPass) {
        const bloomPass = window.werewolfThreeJs.demo._bloomPass;
        
        switch(type) {
            case 'strength':
                bloomPass.strength = floatValue;
                break;
            case 'radius':
                bloomPass.radius = floatValue;
                break;
            case 'threshold':
                bloomPass.threshold = floatValue;
                break;
        }
        
        console.debug(`[Sky Controls] Updated bloom ${type} to ${floatValue.toFixed(2)}`);
    }
}

function updateClouds(type, value) {
    const floatValue = parseFloat(value);
    const valueId = `sky-cloud-${type}-value`;
    const valueEl = document.getElementById(valueId);
    if (valueEl) valueEl.textContent = floatValue.toFixed(1);
    
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._clouds) {
        const clouds = window.werewolfThreeJs.demo._clouds;
        
        clouds.forEach(cloud => {
            if (!cloud || !cloud.material) return;
            
            switch(type) {
                case 'opacity':
                    cloud.material.opacity = floatValue;
                    break;
                case 'speed':
                    cloud.userData = cloud.userData || {};
                    cloud.userData.speed = floatValue * 0.0001;
                    break;
            }
        });
        
        console.debug(`[Sky Controls] Updated cloud ${type} to ${floatValue.toFixed(1)}`);
    }
}

function toggleClouds(enabled) {
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._clouds) {
        window.werewolfThreeJs.demo._clouds.forEach(cloud => {
            if (cloud) cloud.visible = enabled;
        });
        console.debug(`[Sky Controls] Clouds ${enabled ? 'enabled' : 'disabled'}`);
    }
}

function updateGodRays(type, value) {
    const floatValue = parseFloat(value);
    const valueEl = document.getElementById('sky-godray-intensity-value');
    if (valueEl) valueEl.textContent = floatValue.toFixed(1);
    
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
        const demo = window.werewolfThreeJs.demo;
        
        if (type === 'intensity') {
            if (demo.setGodRayIntensity) {
                demo.setGodRayIntensity(floatValue);
            } else if (demo._godRayIntensity !== undefined) {
                demo._godRayIntensity = floatValue;
            }
        }
        
        console.debug(`[Sky Controls] Updated god ray ${type} to ${floatValue.toFixed(1)}`);
    }
}

function toggleGodRays(enabled) {
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
        const demo = window.werewolfThreeJs.demo;
        const intensityEl = document.getElementById('sky-godray-intensity');
        const intensity = intensityEl ? parseFloat(intensityEl.value) : 1.0;
        
        if (enabled) {
            demo._godRayIntensity = intensity;
        } else {
            demo._godRayIntensity = 0;
        }
        console.debug(`[Sky Controls] God rays ${enabled ? 'enabled' : 'disabled'}`);
    }
}

function updateDayNightLighting(elevation) {
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
        const demo = window.werewolfThreeJs.demo;
        const isDay = elevation > 0;
        
        // Update phase indicator
        const phaseIndicator = document.getElementById('sky-current-phase');
        if (phaseIndicator) {
            phaseIndicator.textContent = isDay ? 'DAY' : 'NIGHT';
            phaseIndicator.className = `sky-phase-indicator ${isDay ? 'sky-phase-day' : 'sky-phase-night'}`;
        }
        
        // Trigger phase update if available
        if (demo.updateSkyForPhase) {
            demo.updateSkyForPhase(isDay);
        }
    }
}

function setSkyDayTime() {
    console.debug('[Sky Controls] Setting day time');
    const elevationEl = document.getElementById('sky-elevation');
    const azimuthEl = document.getElementById('sky-azimuth');
    const timeEl = document.getElementById('sky-time-slider');
    
    if (elevationEl) elevationEl.value = 45;
    if (azimuthEl) azimuthEl.value = 180;
    if (timeEl) timeEl.value = 12;
    
    updateSunPosition('elevation', 45);
    updateSunPosition('azimuth', 180);
    setTimeOfDay(12);
}

function setSkyNightTime() {
    console.debug('[Sky Controls] Setting night time');
    const elevationEl = document.getElementById('sky-elevation');
    const azimuthEl = document.getElementById('sky-azimuth');
    const timeEl = document.getElementById('sky-time-slider');
    
    if (elevationEl) elevationEl.value = -45;
    if (azimuthEl) azimuthEl.value = 0;
    if (timeEl) timeEl.value = 0;
    
    updateSunPosition('elevation', -45);
    updateSunPosition('azimuth', 0);
    setTimeOfDay(0);
}

function setTimeOfDay(value) {
    const hours = parseFloat(value);
    const h = Math.floor(hours);
    const m = Math.floor((hours - h) * 60);
    
    const timeValueEl = document.getElementById('sky-time-value');
    if (timeValueEl) {
        timeValueEl.textContent = `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
    }
    
    if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._sky) {
        // Convert hours to sun position
        const sunAngle = (hours / 24) * Math.PI * 2 - Math.PI / 2;
        const elevation = Math.sin(sunAngle) * 90;
        const azimuth = (hours / 24) * 360;
        
        // Update sliders
        const elevationEl = document.getElementById('sky-elevation');
        const azimuthEl = document.getElementById('sky-azimuth');
        if (elevationEl) elevationEl.value = elevation;
        if (azimuthEl) azimuthEl.value = azimuth;
        
        // Update sun position
        updateSunPosition('elevation', elevation);
        updateSunPosition('azimuth', azimuth);
        
        console.debug(`[Sky Controls] Time updated to ${h}:${m.toString().padStart(2, '0')}`);
    }
    
    updateSkyInfo();
}

let skyTransitionInterval = null;
function toggleSkyTransition() {
    if (skyTransitionInterval) {
        clearInterval(skyTransitionInterval);
        skyTransitionInterval = null;
        console.debug('[Sky Controls] Auto transition stopped');
    } else {
        console.debug('[Sky Controls] Auto transition started');
        skyTransitionInterval = setInterval(() => {
            const timeEl = document.getElementById('sky-time-slider');
            if (timeEl) {
                let currentTime = parseFloat(timeEl.value);
                currentTime += 0.1;
                if (currentTime >= 24) currentTime = 0;
                timeEl.value = currentTime;
                setTimeOfDay(currentTime);
            }
        }, 100);
    }
}

function updateSkyInfo() {
    if (!window.werewolfThreeJs || !window.werewolfThreeJs.demo) return;
    
    const demo = window.werewolfThreeJs.demo;
    
    if (demo._sky && demo._sky.material && demo._sky.material.uniforms) {
        const sunPos = demo._sky.material.uniforms['sunPosition'].value;
        
        // Calculate sun angles
        const elevation = Math.asin(sunPos.y) * (180 / Math.PI);
        const azimuth = Math.atan2(sunPos.x, sunPos.z) * (180 / Math.PI);
        
        const sunPosEl = document.getElementById('sky-sun-position');
        if (sunPosEl) {
            sunPosEl.textContent = `Az: ${azimuth.toFixed(1)}°, El: ${elevation.toFixed(1)}°`;
        }
        
        // Moon position (opposite of sun)
        const moonPosEl = document.getElementById('sky-moon-position');
        if (moonPosEl) {
            if (elevation < 0) {
                const moonElevation = -elevation;
                const moonAzimuth = (azimuth + 180) % 360;
                moonPosEl.textContent = `Az: ${moonAzimuth.toFixed(1)}°, El: ${moonElevation.toFixed(1)}°`;
            } else {
                moonPosEl.textContent = 'Hidden';
            }
        }
    }
}

// Update info periodically
setInterval(updateSkyInfo, 500);
