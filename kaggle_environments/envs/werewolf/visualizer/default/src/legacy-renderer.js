import { ThreeModules } from './world/ThreeLoader.js';
import { World } from './world/World.js';
import {
  playAudioFrom,
  stopAndClearAudio,
  setAudioContext
} from './audio/AudioController.js';
import {
  createPlayerIdReplacer,
  updateEventLog,
  FALLBACK_THUMBNAIL_IMG,
  shuffleIds
} from './utils/helpers.js';
import { disambiguateDisplayNames, simplifyDisplayNames } from './utils/nameUtils.js';
import { updateSkyInfo } from './ui/SkyControls.js';

if (!window.werewolfThreeJs) {
  window.werewolfThreeJs = {
    initialized: false,
    world: null,
    players3DInitialized: false,
    resizeObserver: null,
  };
}
const threeState = window.werewolfThreeJs;

// Add event listeners for custom audio events
window.addEventListener('audio-toggle', (e) => {
    const audioState = window.kaggleWerewolf;
    if (audioState) {
        audioState.isAudioEnabled = e.detail.enabled;
        if (!e.detail.enabled) {
            stopAndClearAudio(audioState);
            audioState.isPaused = true;
        } else {
            // Logic to resume or start handled in playAudioFrom/playNextInQueue if pending
        }
    }
});

window.addEventListener('audio-speed', (e) => {
    const audioState = window.kaggleWerewolf;
    if (audioState) {
        audioState.playbackRate = e.detail.rate;
        if (audioState.isAudioPlaying && audioState.audioPlayer) {
            audioState.audioPlayer.playbackRate = e.detail.rate;
        }
    }
});

// Listen for speed changes from the ReplayVisualizer (parent controls)
window.addEventListener('replayer-speed', (e) => {
    // Skip if this event originated from the ReplayVisualizer itself
    if (e.detail.fromReplayer) {
        const audioState = window.kaggleWerewolf;
        if (audioState) {
            audioState.playbackRate = e.detail.rate;
            if (audioState.isAudioPlaying && audioState.audioPlayer) {
                audioState.audioPlayer.playbackRate = e.detail.rate;
            }
        }
    }
});

export function renderer(context, parent) {
  const { replay, step, height = 1000, width = 1500 } = context;
  const environment = replay;
  const parentId = parent.id;

  if (window.kaggleWerewolf && replay && replay.info) {
    window.kaggleWerewolf.episodeId = replay.info.EpisodeId;
  }

  let playerNamesFor3D = [];
  let playerThumbnailsFor3D = {};

  // Set audio context immediately
  // POlyfill/Wrap setCurrentStep to handle audio-driven updates without stopping playback
  // We MUST use direct synchronous control (unstable_replayerControls) if available,
  // because the default postMessage approach is async and causes our isSystemMove flag
  // to reset before the step update actually happens.
  const originalSetCurrentStep = context.setCurrentStep;
  context.setCurrentStep = (s) => {
    if (window.kaggleWerewolf) window.kaggleWerewolf.isSystemMove = true;
    try {
      if (context.unstable_replayerControls) {
        context.unstable_replayerControls.setStep(s);
      } else if (originalSetCurrentStep) {
        originalSetCurrentStep(s);
      }
    } finally {
      if (window.kaggleWerewolf) window.kaggleWerewolf.isSystemMove = false;
    }
  };

  // Override setPlaying to update the Visualizer UI state (Play/Pause button)
  // The default implementation only posts a message which the visualizer ignores.
  if (context.unstable_replayerControls && context.unstable_replayerControls.setPlaying) {
    const originalSetPlaying = context.setPlaying;
    context.setPlaying = (playing) => {
      // Update UI
      context.unstable_replayerControls.setPlaying(playing);
      // Call original (postMessage) just in case
      if (originalSetPlaying) originalSetPlaying(playing);
    };
  }
  setAudioContext(context);

  const systemEntryTypeSet = new Set([
    'moderator_announcement', 'elimination', 'vote_request', 'heal_request',
    'heal_result', 'inspect_request', 'inspect_result', 'bidding_info',
    'bid_result', 'day_start', 'night_start',
  ]);

  // Initialize or update werewolfGamePlayer when visualizerData is available
  // We check if allEvents length differs to detect when a new replay is loaded
  const shouldInitPlayer = !window.werewolfGamePlayer ||
    (environment.visualizerData &&
     window.werewolfGamePlayer.allEvents?.length !== environment.visualizerData.allEvents?.length);

  if (shouldInitPlayer) {
    if (environment.visualizerData) {
        const vData = environment.visualizerData;
        window.werewolfGamePlayer = {
            initialized: false,
            allEvents: vData.allEvents,
            displayEvents: [],
            eventToKaggleStep: vData.eventToKaggleStep,
            displayStepToAllEventsIndex: vData.displayStepToAllEventsIndex,
            allEventsIndexToDisplayStep: vData.allEventsIndexToDisplayStep,
            originalSteps: vData.originalSteps,
            reasoningCounter: 0,
        };
        // Reconstruct displayEvents for internal use
        window.werewolfGamePlayer.displayEvents = vData.displayStepToAllEventsIndex.map(
            (idx) => vData.allEvents[idx]
        );
      window.wwCurrentStep = step || 0;
    } else {
        console.warn("Visualizer Data not found. Ensure the transformer is processing the replay.");
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
      window.wwCurrentStep = step || 0;
    }
    window.werewolfGamePlayer.initialized = true;
  }

  // --- Audio State ---
  const audioMap = window.AUDIO_MAP || {};
  if (!window.kaggleWerewolf) {
    window.kaggleWerewolf = {
      audioQueue: [],
      isAudioPlaying: false,
      isAudioEnabled: false,
      isPaused: false,
      lastPlayedStep: parseInt(sessionStorage.getItem('ww_lastPlayedStep') || '-1', 10),
      audioPlayer: new Audio(),
      playbackRate: 1,
      allEvents: null,
      audioContextActivated: false,
    };
  }
  const audioState = window.kaggleWerewolf;
  if (audioState.hasAudioTracks === undefined) {
    audioState.hasAudioTracks = Object.keys(audioMap).length > 0;
  }

  // --- Patch Controls ---
  // Track which instance we patched to handle HMR correctly
  const playerInstance = context.unstable_replayerControls?._replayerInstance;
  const shouldPatchControls = playerInstance &&
    window.werewolfPatchedInstance !== playerInstance;

  if (shouldPatchControls) {
    window.werewolfPatchedInstance = playerInstance;
    const originalSetStep = playerInstance.setStep.bind(playerInstance);
    const originalPlay = playerInstance.play.bind(playerInstance);
    const originalPause = playerInstance.pause.bind(playerInstance);

    playerInstance.setStep = (newStep) => {
      // Only stop audio if this is a USER interaction (not a system auto-advance)
      if (!audioState.isSystemMove) {
        stopAndClearAudio(audioState, parentId);
        audioState.isPaused = true;
      }
      const currentStep = window.wwCurrentStep || 0;
      if (newStep !== currentStep + 1) {
        resetThreeJsState();
      }
      window.wwCurrentStep = newStep;
      originalSetStep(newStep);
    };

    playerInstance.play = (continuing) => {
      if (audioState.isAudioEnabled) {
        originalPause();
        context.setPlaying(true); // Use adapter interface
        // FIX: Use window.wwCurrentStep to get the LIVE step, not context.step (which is static)
        let currentDisplayStep = window.wwCurrentStep || 0;
        const newStepsLength = window.werewolfGamePlayer.displayEvents.length;
        if (!continuing && !audioState.isPaused && currentDisplayStep === newStepsLength - 1) {
          currentDisplayStep = 0;
          originalSetStep(0);
          window.wwCurrentStep = 0; // Ensure sync
        }
        const allEventsIndex = window.werewolfGamePlayer.displayStepToAllEventsIndex[currentDisplayStep];
        if (allEventsIndex === undefined) {
          context.setPlaying(false); // Use adapter interface
          return;
        }
        playAudioFrom(allEventsIndex, true);
      } else {
        originalPlay(continuing);
      }
    };

    playerInstance.pause = () => {
      originalPause();
      context.setPlaying(false); // Use adapter interface
      audioState.isPaused = true;
      if (audioState.isAudioPlaying) {
        audioState.audioPlayer.pause();
      }
    };
  }

  // --- UI Setup ---
  let mainContainer = parent.querySelector('.main-container');

  // --- Initialize 3D World ---
  function initThreeJs() {
    // Get actual dimensions from the parent element for responsive sizing
    const rect = parent.getBoundingClientRect();
    const actualWidth = rect.width || width;
    const actualHeight = rect.height || height;

    if (threeState.initialized) {
        // Re-attach canvas if needed (e.g. if parent changed)
        if (threeState.world && threeState.world.sceneManager.renderer.domElement && !parent.contains(threeState.world.sceneManager.renderer.domElement)) {
            parent.appendChild(threeState.world.sceneManager.renderer.domElement);
            parent.appendChild(threeState.world.sceneManager.labelRenderer.domElement);
        }

        // Handle Resize - use actual parent dimensions
        if (threeState.world && (threeState.world.options.width !== actualWidth || threeState.world.options.height !== actualHeight)) {
            threeState.world.resize(actualWidth, actualHeight);
        }
        return;
    }

    try {
        threeState.world = new World({ parent, width: actualWidth, height: actualHeight }, ThreeModules);
        threeState.initialized = true;
        window.werewolfThreeJs.demo = threeState.world;
        if (playerNamesFor3D.length > 0 && !threeState.players3DInitialized) {
            setup3DPlayers();
        }

        // Set up ResizeObserver to handle window/container resize
        if (!threeState.resizeObserver) {
            threeState.resizeObserver = new ResizeObserver((entries) => {
                for (const entry of entries) {
                    if (threeState.world && entry.contentRect) {
                        const newWidth = entry.contentRect.width;
                        const newHeight = entry.contentRect.height;
                        if (newWidth > 0 && newHeight > 0) {
                            threeState.world.resize(newWidth, newHeight);
                        }
                    }
                }
            });
            threeState.resizeObserver.observe(parent);
        }
    } catch (err) {
        console.error("Failed to initialize 3D world", err);
        parent.textContent = "Error loading 3D assets.";
    }
}

  initThreeJs();

  if (!environment || !environment.steps || environment.steps.length === 0 || step >= environment.steps.length) {
    if (!mainContainer) {
      const tempContainer = document.createElement('div');
      tempContainer.textContent = 'Waiting for game data or invalid step...';
      parent.appendChild(tempContainer);
    }
    return;
  }

  const player = window.werewolfGamePlayer;
  const { allEvents, displayStepToAllEventsIndex, originalSteps, eventToKaggleStep } = player;

  if (step >= displayStepToAllEventsIndex.length) return;
  const allEventsIndex = displayStepToAllEventsIndex[step];
  const eventStep = allEventsIndex;
  const kaggleStep = eventToKaggleStep[eventStep] || 0;

  // Track last update time to detect playback
  if (window.werewolfThreeJs) {
    window.werewolfThreeJs.lastStepUpdateTime = performance.now();
  }

  let gameState = {
    players: [],
    day: 0,
    phase: 'GAME_SETUP',
    game_state_phase: 'DAY',
    gameWinner: null,
    eventLog: [],
    playerThreatLevels: new Map(),
  };

  const agentConfigMap = new Map();
  if (environment.configuration && environment.configuration.agents) {
    environment.configuration.agents.forEach((agent) => {
      if (agent && agent.id) agentConfigMap.set(agent.id, agent);
    });
  }

  // Override/Supplement with GAME_END info if available (GAME_END is ground truth, since config maybe overridden)
  if (environment.info && environment.info.GAME_END && environment.info.GAME_END.all_players) {
    environment.info.GAME_END.all_players.forEach((p) => {
      if (p.agent && p.agent.id) {
        const existing = agentConfigMap.get(p.agent.id) || {};
        agentConfigMap.set(p.agent.id, { ...existing, ...p.agent });
      }
    });
  }

  const firstObs = originalSteps[0]?.[0]?.observation?.raw_observation;
  let allPlayerNamesList = [];
  let playerThumbnails = {};

  // Try to find player list from Moderator Observation (fallback if raw_observation is missing)
  let moderatorPlayerIds = null;
  if (environment.info && environment.info.MODERATOR_OBSERVATION && environment.info.MODERATOR_OBSERVATION[0]) {
    const firstModStep = environment.info.MODERATOR_OBSERVATION[0];
    for (const entry of firstModStep) {
      if (entry.data_type === 'GameStartDataEntry') {
        try {
          const parsed = JSON.parse(entry.json_str);
          if (parsed.data && parsed.data.player_ids) {
            moderatorPlayerIds = parsed.data.player_ids;
            break;
          }
        } catch (e) {
          console.error("Error parsing GameStartDataEntry", e);
        }
      }
    }
  }

  if (firstObs && firstObs.all_player_ids) {
    allPlayerNamesList = firstObs.all_player_ids;
    playerThumbnails = firstObs.player_thumbnails || {};
  } else if (moderatorPlayerIds) {
    allPlayerNamesList = moderatorPlayerIds;
  } else if (environment.configuration && environment.configuration.randomize_ids && environment.configuration.seed) {
    // Deterministic fallback using the same LCG shuffle as python engine
    allPlayerNamesList = shuffleIds(environment.configuration.agents, environment.configuration.seed);
  } else {
    allPlayerNamesList = Array.from(agentConfigMap.keys());
  }

  playerNamesFor3D = [...allPlayerNamesList];
  playerThumbnailsFor3D = { ...playerThumbnails };
  allPlayerNamesList.forEach(id => {
      const conf = agentConfigMap.get(id);
      if (conf && conf.thumbnail) {
          playerThumbnailsFor3D[id] = conf.thumbnail;
      }
  });

  if (!allPlayerNamesList || allPlayerNamesList.length === 0) {
      // Waiting
      return;
  }

  gameState.players = allPlayerNamesList.map((playerId) => {
    const configAgent = agentConfigMap.get(playerId) || {};
    // Prioritize configAgent.thumbnail (from GAME_END or config) over initial observation
    const thumbnail = configAgent.thumbnail || playerThumbnails[playerId] || FALLBACK_THUMBNAIL_IMG;
    return {
      name: playerId,
      is_alive: true,
      role: 'Unknown',
      team: 'Unknown',
      status: 'Alive',
      thumbnail: thumbnail,
      display_name: configAgent.display_name || playerId,
    };
  });

  simplifyDisplayNames(gameState.players);
  disambiguateDisplayNames(gameState.players);

  const playerMap = new Map(gameState.players.map((p) => [p.name, p]));

  if (!player.playerIdReplacer) {
    player.playerIdReplacer = createPlayerIdReplacer(playerMap);
  }

  gameState.players.forEach((p) => gameState.playerThreatLevels.set(p.name, 0));

  const moderatorInitialLog = environment.info?.MODERATOR_OBSERVATION?.[0] || [];
  moderatorInitialLog.flat().forEach((dataEntry) => {
    if (dataEntry.data_type === 'GameStartRoleDataEntry') {
      const historyEvent = JSON.parse(dataEntry.json_str);
      const data = historyEvent.data;
      if (data) {
          const p = playerMap.get(data.player_id);
          if (p) { p.role = data.role; p.team = data.team; }
      }
    }
  });

  function threatStringToLevel(threatString) {
    switch (threatString) {
      case 'SAFE': return 0;
      case 'UNEASY': return 0.5;
      case 'DANGER': return 1.0;
      default: return 0;
    }
  }

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
                gameState.eventLog.push({ type: 'timeout', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, actor_id: match[1], reasoning: 'Timed out', timestamp: historyEvent.created_at });
            }
        } else if (historyEvent.event_name === 'day_start' || historyEvent.event_name === 'night_start') {
            gameState.eventLog.push({ type: 'system', step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, text: historyEvent.description, allEventsIndex: i, timestamp });
        }
        continue;
    }

    // Process event types (simplified mapping)
    const commonProps = { step: historyEvent.kaggleStep, day: historyEvent.day, phase: historyEvent.phase, allEventsIndex: i, timestamp, event_name: historyEvent.event_name };
    
    if (historyEvent.dataType === 'ChatDataEntry') {
        gameState.eventLog.push({ type: 'chat', ...commonProps, actor_id: data.actor_id, speaker: data.actor_id, message: data.message, reasoning: data.reasoning, mentioned_player_ids: data.mentioned_player_ids || [] });
    } else if (historyEvent.dataType === 'DayExileVoteDataEntry') {
        gameState.eventLog.push({ type: 'vote', ...commonProps, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning });
    } else if (historyEvent.dataType === 'WerewolfNightVoteDataEntry') {
        gameState.eventLog.push({ type: 'night_vote', ...commonProps, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning });
    } else if (historyEvent.dataType === 'DoctorHealActionDataEntry') {
        gameState.eventLog.push({ type: 'doctor_heal_action', ...commonProps, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning });
    } else if (historyEvent.dataType === 'SeerInspectActionDataEntry') {
        gameState.eventLog.push({ type: 'seer_inspection', ...commonProps, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning });
    } else if (historyEvent.dataType === 'DayExileElectedDataEntry') {
        gameState.eventLog.push({ type: 'exile', ...commonProps, name: data.elected_player_id, role: data.elected_player_role_name });
    } else if (historyEvent.dataType === 'WerewolfNightEliminationDataEntry') {
        gameState.eventLog.push({ type: 'elimination', ...commonProps, name: data.eliminated_player_id, role: data.eliminated_player_role_name });
    } else if (historyEvent.dataType === 'SeerInspectResultDataEntry') {
        gameState.eventLog.push({ type: 'seer_inspection_result', ...commonProps, actor_id: data.actor_id, seer: data.actor_id, target: data.target_id, role: data.role, team: data.team });
    } else if (historyEvent.dataType === 'DoctorSaveDataEntry') {
        gameState.eventLog.push({ type: 'save', ...commonProps, saved_player: data.saved_player_id });
    } else if (historyEvent.dataType === 'PhaseDividerDataEntry') {
        gameState.eventLog.push({ type: 'phase_divider', ...commonProps, divider: data.divider_type });
    } else if (historyEvent.dataType === 'GameEndResultsDataEntry') {
        gameState.gameWinner = data.winner_team;
        const winners = gameState.players.filter((p) => p.team === data.winner_team).map((p) => p.name);
        const losers = gameState.players.filter((p) => p.team !== data.winner_team).map((p) => p.name);
        gameState.eventLog.push({ type: 'game_over', ...commonProps, day: Infinity, phase: 'GAME_OVER', winner: data.winner_team, winners, losers });
    } else if (historyEvent.dataType === 'DiscussionOrderDataEntry') {
        gameState.eventLog.push({ type: 'system', ...commonProps, text: historyEvent.description });
    } else if (systemEntryTypeSet.has(historyEvent.event_name)) {
        gameState.eventLog.push({ type: 'system', ...commonProps, text: historyEvent.description, data: data });
    }
  }

  // Audio Cleanup
  if (eventStep < audioState.lastPlayedStep) {
    audioState.audioQueue = [];
    audioState.isAudioPlaying = false;
    if (audioState.audioPlayer) audioState.audioPlayer.pause();
    const chatLog = parent.querySelector('#chat-log');
    if (chatLog) chatLog.innerHTML = '';
  }
  audioState.lastPlayedStep = eventStep;
  sessionStorage.setItem('ww_lastPlayedStep', eventStep);

  // Update Player Status based on Log
  gameState.players.forEach((p) => { p.is_alive = true; p.status = 'Alive'; });
  gameState.eventLog.forEach((entry) => {
    if (entry.type === 'exile' || entry.type === 'elimination') {
      const player = playerMap.get(entry.name);
      if (player) {
        player.is_alive = false;
        player.status = entry.type === 'exile' ? 'Exiled' : 'Eliminated';
      }
    }
  });

  const currentEvent = allEvents[eventStep];
  let nameToHighlight = null;
  if (currentEvent) {
    if (currentEvent.data && currentEvent.data.actor_id) {
      nameToHighlight = currentEvent.data.actor_id;
    } else if (currentEvent.event_name === 'vote_action' && !currentEvent.data) {
      const match = currentEvent.description.match(/P(player_\d+)/);
      if (match && playerMap.has(match[1])) nameToHighlight = match[1];
    }
  }

  // Use CSS-based sizing for responsive behavior instead of fixed pixel dimensions
  // The .werewolf-parent class uses width: 100%; height: 100%; which works with flexbox
  parent.classList.add('werewolf-parent');

  if (!mainContainer) {
    mainContainer = document.createElement('div');
    mainContainer.className = 'main-container';
    parent.appendChild(mainContainer);
  }

  // Define speak function to be passed to UI
  const speak = (index) => {
      // 1. Find the corresponding display step
      const displayStep = window.werewolfGamePlayer.allEventsIndexToDisplayStep[index];
      // 2. Jump slider
      if (displayStep !== undefined && context.unstable_replayerControls) {
          context.unstable_replayerControls.setStep(displayStep);
      }
      // 3. Play
      playAudioFrom(index, true);
  };

  // Update UI Panels
  updateUIPanels(parent, mainContainer, gameState, allEvents[eventStep], playerMap, speak);

  // 3D Initialization logic
  function setup3DPlayers() {
      if (!threeState.world) return;
      threeState.players3DInitialized = true;
      threeState.world.characterManager.initializePlayers(gameState, playerNamesFor3D, playerThumbnailsFor3D, threeState.world.uiManager)
        .then(() => {
            updateSceneFromGameState(gameState, playerMap, nameToHighlight);
        })
        .catch(e => {
            console.error("Error initializing players", e);
            threeState.players3DInitialized = false;
        });
  }

  // If world is ready, trigger player init or update scene
  if (threeState.world) {
      if (!threeState.players3DInitialized && playerNamesFor3D.length > 0) {
          setup3DPlayers();
      } else if (threeState.players3DInitialized) {
          updateSceneFromGameState(gameState, playerMap, nameToHighlight);
      }
  }

  function resetThreeJsState() {
      if (!threeState.world) return;
      threeState.world.characterManager.reset();
      threeState.world.voteVisuals.updateVoteVisuals(new Map(), threeState.world.characterManager.playerObjects, true);
  }

  function updateSceneFromGameState(gameState, playerMap, actingPlayerName) {
      if (!threeState.world) return;
      const world = threeState.world;
      
      // Ensure we start with a clean animation state for this step
      world.characterManager.resetAnimations();
      
      // Hide chat bubbles
      world.characterManager.playerObjects.forEach(p => {
          if (p.playerUI && p.playerUI.element) p.playerUI.element.classList.remove('chat-active');
      });

      const logUpToCurrentStep = gameState.eventLog;
      const lastEvent = logUpToCurrentStep.length > 0 ? logUpToCurrentStep[logUpToCurrentStep.length - 1] : null;
      
      let phase = gameState.game_state_phase;
      if (lastEvent && lastEvent.phase) phase = lastEvent.phase;

      // Update Player Statuses
      gameState.players.forEach(player => {
          const threatLevel = gameState.playerThreatLevels.get(player.name) || 0;
          let primaryStatus = 'default';
          if (!player.is_alive) primaryStatus = 'dead';
          else if (player.role === 'Werewolf' && phase.toUpperCase() === 'NIGHT') primaryStatus = 'werewolf';
          else if (player.role === 'Doctor' && phase.toUpperCase() === 'NIGHT') primaryStatus = 'doctor';
          else if (player.role === 'Seer' && phase.toUpperCase() === 'NIGHT') primaryStatus = 'seer';

          const justDied = lastEvent && (lastEvent.type === 'exile' || lastEvent.type === 'elimination') && lastEvent.name === player.name;
          world.characterManager.updatePlayerStatus(player.name, player, primaryStatus, threatLevel, justDied);
      });

      const currentEventIndex = lastEvent ? lastEvent.allEventsIndex : 0;
      world.updatePhase(phase, currentEventIndex);

      // Voting Visuals
      const currentVotes = new Map();
      const lastNightStart = logUpToCurrentStep.findLastIndex(e => e.type === 'phase_divider' && e.divider === 'NIGHT START');
      const lastDayVoteStart = logUpToCurrentStep.findLastIndex(e => e.type === 'phase_divider' && e.divider === 'DAY VOTE START');
      const sessionStartIndex = Math.max(lastNightStart, lastDayVoteStart);
      
      let isVotingSession = false;
      if (sessionStartIndex > -1) {
          const lastOutcomeEventIndex = logUpToCurrentStep.findLastIndex(e => e.type === 'exile' || e.type === 'elimination' || e.type === 'save');
          if (sessionStartIndex > lastOutcomeEventIndex || (lastOutcomeEventIndex > -1 && lastOutcomeEventIndex === logUpToCurrentStep.length - 1)) {
              isVotingSession = true;
          }
      }

      if (isVotingSession) {
          const alivePlayerNames = new Set(gameState.players.filter(p => p.is_alive).map(p => p.name));
          const relevantEvents = logUpToCurrentStep.slice(sessionStartIndex);
          for (const event of relevantEvents) {
              if (['vote', 'night_vote', 'doctor_heal_action', 'seer_inspection'].includes(event.type)) {
                  if (alivePlayerNames.has(event.actor_id)) {
                      currentVotes.set(event.actor_id, { target: event.target, type: event.type });
                  }
              } else if (event.type === 'timeout') {
                  currentVotes.delete(event.actor_id);
              }
          }
      }
      world.voteVisuals.updateVoteVisuals(currentVotes, world.characterManager.playerObjects, !isVotingSession);

      // Action Animations
      let subtitleShown = false;
      if (lastEvent) {
          let messageForBubble = '';
          let reasoningForBubble = lastEvent.reasoning || '';
          const actorName = lastEvent.actor_id || lastEvent.speaker;

          switch (lastEvent.type) {
              case 'chat':
                  messageForBubble = `"${lastEvent.message}"`;
                  world.characterManager.triggerSpeakingAnimation(actorName);
                  break;
              case 'vote':
              case 'night_vote':
                  messageForBubble = `Votes for <strong>${lastEvent.target}</strong>.`;
                  world.characterManager.triggerPointingAnimation(actorName, lastEvent.target);
                  break;
              case 'doctor_heal_action':
                  messageForBubble = `Heals <strong>${lastEvent.target}</strong>.`;
                  world.characterManager.triggerPointingAnimation(actorName, lastEvent.target);
                  break;
              case 'seer_inspection':
                  messageForBubble = `Inspects <strong>${lastEvent.target}</strong>.`;
                  world.characterManager.triggerPointingAnimation(actorName, lastEvent.target);
                  break;
              case 'system':
                  // NEW: Display moderator announcement
                  let announcement = lastEvent.text;
                  if (window.werewolfGamePlayer && window.werewolfGamePlayer.playerIdReplacer) {
                      announcement = window.werewolfGamePlayer.playerIdReplacer(announcement);
                  }
                  world.uiManager.displayModeratorAnnouncement(announcement);
                  subtitleShown = true;
                  break;
          }

          if (messageForBubble && actorName && playerMap.has(actorName)) {
              const formattedMessage = window.werewolfGamePlayer.playerIdReplacer(messageForBubble);
              const playerObj = world.characterManager.playerObjects.get(actorName);
              if (playerObj && playerObj.playerUI) {
                  world.uiManager.displayPlayerBubble(playerObj.playerUI, formattedMessage, reasoningForBubble, lastEvent.timestamp);
                  world.characterManager.updatePlayerActive(actorName);
                  subtitleShown = true;
              }
          }

          if (lastEvent.type === 'game_over') {
              if (lastEvent.winners) lastEvent.winners.forEach(w => { if (playerMap.has(w)) world.characterManager.triggerVictoryAnimation(w); });
              if (lastEvent.losers) lastEvent.losers.forEach(l => { if (playerMap.has(l)) world.characterManager.triggerDefeatedAnimation(l); });
          } else if (lastEvent.event_name === 'moderator_announcement') {
              gameState.players.forEach(p => { if (p.is_alive) world.characterManager.updatePlayerActive(p.name); });
          } else if (lastEvent.actor_id && playerMap.has(lastEvent.actor_id)) {
              world.characterManager.updatePlayerActive(lastEvent.actor_id);
          }
      }
      
    // STICKY SUBTITLES:
    // If no new subtitle was shown this step, check if we should clear the old one.
    // We ONLY clear if the current event is effectively a "speech" event that happened to be empty
    // (which shouldn't happen often) or if we explicitly want silence.
    // However, for "System" events like phase dividers that are effectively silent,
    // we want to PERSIST the last message (e.g. Moderator info) so the user has time to read it.

      if (!subtitleShown) {
        // Check if the current event is a "Silent" system event (e.g. phase divider, day start)
        // If it IS silent, we do NOT clear. We let the previous message stick.
        const silentEventTypes = ['phase_divider', 'day_start', 'night_start', 'vote_request', 'heal_request', 'inspect_request'];
        const isSilent = lastEvent && (silentEventTypes.includes(lastEvent.type) || silentEventTypes.includes(lastEvent.event_name));

        // If it's NOT a silent event (meaning it probably SHOULD have shown something but didn't, or it's a new turn),
        // or if we have changed PHASE significantly (though usually phase change has a divider),
        // AND we aren't in a "sticky" state... 
        // Actually, simplest logic for better UX: 
        // Only clear if we have a NEW active event that replaces the visuals but has no text (rare).
        // OR if we explicitly want to clear. 
        // For now, let's just NOT clear on silent events.

        if (!isSilent) {
            world.uiManager.clearSubtitle();
          }
      }
  }
}

function updateUIPanels(parent, mainContainer, gameState, currentEvent, playerMap, onSpeak) {
    let scoreboard = parent.querySelector('.game-scoreboard');
    if (!scoreboard) {
        scoreboard = document.createElement('div');
        scoreboard.className = 'game-scoreboard';
        parent.appendChild(scoreboard);
    }

    const alivePlayers = gameState.players.filter((p) => p.is_alive).length;
    const deadPlayers = gameState.players.filter((p) => !p.is_alive).length;
    const werewolves = gameState.players.filter((p) => p.is_alive && p.role === 'Werewolf').length;
    const villagers = gameState.players.filter((p) => p.is_alive && p.role !== 'Werewolf' && p.role !== 'Unknown').length;

    let currentAction = 'Waiting...';
    const lastEvent = gameState.eventLog[gameState.eventLog.length - 1];
    const isNight = (currentEvent.phase || 'DAY').toUpperCase() === 'NIGHT';

    if (gameState.gameWinner) currentAction = `${gameState.gameWinner} Win!`;
    else if (gameState.phase === 'VOTING') currentAction = 'Voting Phase';
    else if (gameState.phase === 'DISCUSSION') currentAction = 'Discussion';
    else if (isNight) {
        if (lastEvent) {
            if (lastEvent.type === 'night_vote') currentAction = 'Werewolves Voting';
            else if (lastEvent.type === 'doctor_heal_action') currentAction = 'Doctor Saving';
            else if (lastEvent.type === 'seer_inspection') currentAction = 'Seer Inspecting';
            else currentAction = 'Night Actions';
        } else currentAction = 'Night Phase';
    } else {
        if (lastEvent && lastEvent.type === 'chat') currentAction = 'Discussion';
        else if (lastEvent && lastEvent.type === 'vote') currentAction = 'Exile Voting';
        else currentAction = 'Day Phase';
    }

    scoreboard.innerHTML = `
        <div class="scoreboard-item"><div id="phase-indicator-capsule" class="phase-indicator"></div></div>
        <div class="scoreboard-item"><div class="scoreboard-label">Alive</div><div class="scoreboard-value alive">${alivePlayers}</div></div>
        <div class="scoreboard-item"><div class="scoreboard-label">Out</div><div class="scoreboard-value dead">${deadPlayers}</div></div>
        ${werewolves > 0 || villagers > 0 ? `
            <div class="scoreboard-item"><div class="scoreboard-label">Werewolves</div><div class="scoreboard-value werewolf">${werewolves}</div></div>
            <div class="scoreboard-item"><div class="scoreboard-label">Villagers</div><div class="scoreboard-value villager">${villagers}</div></div>` : ''}
        <div class="scoreboard-item"><div class="scoreboard-action">${currentAction}</div></div>
    `;

    const phaseCapsule = scoreboard.querySelector('#phase-indicator-capsule');
    if (phaseCapsule) {
        phaseCapsule.className = `phase-indicator ${isNight ? 'night' : 'day'}`;
        const phaseIcon = isNight ? '&#x1F319;' : '&#x2600;';
        phaseCapsule.innerHTML = currentEvent.event_name === 'game_end' ? 
            `<span class="phase-icon">${phaseIcon}</span>` : 
            `<span class="phase-icon">${phaseIcon}</span><span>${currentEvent.day}</span>`;
    }

    let eventPanel = mainContainer.querySelector('.event-panel');
    if (!eventPanel) {
        eventPanel = document.createElement('div');
        eventPanel.className = 'event-panel';
      mainContainer.appendChild(eventPanel);
    }

    updateEventLog(eventPanel, gameState, playerMap, onSpeak);
}

// Update info periodically
setInterval(updateSkyInfo, 500);
