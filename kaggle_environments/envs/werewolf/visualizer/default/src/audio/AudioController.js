let context = null;

export function setAudioContext(ctx) {
  context = ctx;
}

const audioState = window.kaggleWerewolf || {
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

// Ensure it is globally available as legacy renderer expects it
if (!window.kaggleWerewolf) {
  window.kaggleWerewolf = audioState;
}

export function loadQueueFrom(startIndex) {
  console.debug(`DEBUG: [loadQueueFrom] Loading queue from index: ${startIndex}`);
  if (!window.werewolfGamePlayer || !window.werewolfGamePlayer.allEvents) {
      console.error('DEBUG: [loadQueueFrom] CRITICAL: allEvents not found.');
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
              audioEventDetails = {
              message: `${data.actor_id} votes to exile ${data.target_id}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'WerewolfNightVoteDataEntry':
          if (data.actor_id && data.target_id) {
              audioEventDetails = {
              message: `${data.actor_id} votes to eliminate ${data.target_id}.`,
              speaker: 'moderator',
              };
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
              audioEventDetails = {
              message: `${data.elected_player_id} was exiled by vote. Their role was a ${data.elected_player_role_name}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'WerewolfNightEliminationDataEntry':
          if (data.eliminated_player_id && data.eliminated_player_role_name) {
              audioEventDetails = {
              message: `${data.eliminated_player_id} was eliminated. Their role was a ${data.eliminated_player_role_name}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'DoctorSaveDataEntry':
          if (data.saved_player_id) {
              audioEventDetails = {
              message: `${data.saved_player_id} was attacked but saved by a Doctor!`,
              speaker: 'moderator',
              };
          }
          break;
          case 'GameEndResultsDataEntry':
          if (data.winner_team) {
              audioEventDetails = {
              message: `The game is over. The ${data.winner_team} team has won!`,
              speaker: 'moderator',
              };
          }
          break;
          case 'WerewolfNightEliminationElectedDataEntry':
          if (data.elected_target_player_id) {
              audioEventDetails = {
              message: `The werewolves have chosen to eliminate ${data.elected_target_player_id}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'SeerInspectResultDataEntry':
          if (data.role) {
              audioEventDetails = {
              message: `${data.actor_id} saw ${data.target_id}'s role is ${data.role}.`,
              speaker: 'moderator',
              };
          } else if (data.team) {
              audioEventDetails = {
              message: `${data.actor_id} saw ${data.target_id}'s team is ${data.team}.`,
              speaker: 'moderator',
              };
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

export function playAudioFrom(startIndex, isContinuous = true) {
  console.debug(`DEBUG: [playAudioFrom] Called with startIndex: ${startIndex}, isContinuous: ${isContinuous}`);
  if (!audioState.isAudioEnabled) {
    console.error('DEBUG: [playAudioFrom] FAILED: Audio is not enabled.');
    return;
  }

  stopAndClearAudio();
  console.debug('DEBUG: [playAudioFrom] Audio stopped and cleared.');

  if (audioState.isPaused) {
    console.debug('DEBUG: [playAudioFrom] Audio state was paused.');
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
    console.debug('DEBUG: [playAudioFrom] Paused, resuming from same start index (or undefined).');
    // Fall through to load and play.
  }

  audioState.isPaused = false;
  audioState.lastStartedIndex = startIndex;
  loadQueueFrom(startIndex);
  playNextInQueue(isContinuous);
}

export function playNextInQueue(isContinuous = true) {
  if (!context) {
      console.warn("Audio context not set.");
      return;
  }
  
  // Use a global or passed parent ID if possible, but context is safer
  // Assuming context has parent attached? No, context is the kaggle env context
  // We need to find the parent element. 
  // In the original code, parentId was available in scope. 
  // We will assume 'app' or document search, or rely on context having a reference?
  // Actually, let's just look for #chat-log which is unique enough
  const currentParent = document.querySelector('.werewolf-parent') || document.body;

  if (!currentParent) {
    console.error('Werewolf renderer parent container not found in DOM, stopping playback.');
    stopAndClearAudio();
    return;
  }

  console.debug(
    `DEBUG: [playNextInQueue] Called. Queue length: ${audioState.audioQueue.length}. isPaused: ${audioState.isPaused}. isAudioPlaying: ${audioState.isAudioPlaying}.`
  );

  // 1. Clear any previously highlighted element
  const currentlyPlaying = currentParent.querySelector('#chat-log .now-playing');
  if (currentlyPlaying) {
    currentlyPlaying.classList.remove('now-playing');
  }

  if (
    audioState.isPaused ||
    audioState.isAudioPlaying ||
    audioState.audioQueue.length === 0 ||
    !audioState.isAudioEnabled
  ) {
    console.warn(
      `DEBUG: [playNextInQueue] Exiting early. Paused: ${audioState.isPaused}, Playing: ${audioState.isAudioPlaying}, Queue: ${audioState.audioQueue.length}, Enabled: ${audioState.isAudioEnabled}`
    );
    if (audioState.audioQueue.length === 0 && !audioState.isAudioPlaying) {
      console.debug("DEBUG: [playNextInQueue] Playback finished. Setting player to 'paused' state.");
      if (context.setPlaying) {
        context.setPlaying(false);
      }
    }
    return;
  }

  audioState.isAudioPlaying = true;
  const event = audioState.audioQueue.shift();

  // This is the slider logic, it should always run
  if (event.allEventsIndex !== undefined) {
    const displayStep = window.werewolfGamePlayer.allEventsIndexToDisplayStep[event.allEventsIndex];
    console.debug(
      `DEBUG: [playNextInQueue] Found displayStep: ${displayStep} for event index ${event.allEventsIndex}`
    );

    if (displayStep !== undefined && context.setCurrentStep) {
      console.debug(`DEBUG: [playNextInQueue] ### ADVANCING SLIDER TO ${displayStep} ###`);
      context.setCurrentStep(displayStep);

      // Use a short timeout to allow the DOM to update after the step change
      setTimeout(() => {
        const freshParent = document.querySelector('.werewolf-parent') || document.body;
        if (!freshParent) {
          console.error(`DEBUG: Parent element not found after timeout for event index ${event.allEventsIndex}.`);
          return;
        }
        const liToHighlight = freshParent.querySelector(
          `#chat-log li[data-all-events-index="${event.allEventsIndex}"]`
        );
        console.debug(
          `DEBUG: [Timeout] Attempting to highlight element for index ${event.allEventsIndex}`,
          liToHighlight
        );
        if (liToHighlight) {
          liToHighlight.classList.add('now-playing');
          console.debug(
            `DEBUG: [Timeout] Successfully added .now-playing to element for index ${event.allEventsIndex}`
          );
        } else {
          console.error(`DEBUG: [Timeout] FAILED to find element to highlight for index ${event.allEventsIndex}`);
        }
      }, 50); // A small delay to ensure the re-render completes
    } else {
      console.error(
        `DEBUG: [playNextInQueue] CRITICAL: FAILED to advance slider. displayStep: ${displayStep}, playerControls: ${!!context.unstable_replayerControls}`
      );
    }
  }

  const audioMap = window.AUDIO_MAP || {};
  let audioPath = null;
  let audioKey = null;
  if (event.audioEvent) {
    audioKey =
      event.audioEvent.speaker === 'moderator'
        ? `moderator:${event.audioEvent.message}`
        : `${event.audioEvent.speaker}:${event.audioEvent.message}`;
    audioPath = audioMap[audioKey];
  }

  if (audioPath) {
    console.debug(
      `DEBUG: [playNextInQueue] Popped event for index: ${event.allEventsIndex}. Audio key: "${audioKey}"`
    );
    console.debug(`DEBUG: [playNextInQueue] Playing audio: ${audioPath}`);
    audioState.audioPlayer.src = audioPath;
    audioState.audioPlayer.playbackRate = audioState.playbackRate;
    audioState.audioPlayer.onended = () => {
      console.debug(`DEBUG: [onended] Audio for index ${event.allEventsIndex} finished.`);
      audioState.isAudioPlaying = false;
      if (!audioState.isPaused && isContinuous) {
        console.debug('DEBUG: [onended] Calling playNextInQueue recursively.');
        playNextInQueue(isContinuous);
      } else {
        console.debug('DEBUG: [onended] Loop stopped. isPaused or !isContinuous.');
      }
    };
    audioState.audioPlayer.onerror = () => {
      console.error(`DEBUG: [onerror] Audio failed to play for key: "${audioKey}"`);
      audioState.isAudioPlaying = false;
      playNextInQueue(isContinuous);
    };
    audioState.audioPlayer.play().catch((e) => {
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
    }, context ? (context.speed || 1000) : 1000);
  }
}

export function stopAndClearAudio(state = audioState, parentId) {
  if (state.isAudioPlaying) {
    state.audioPlayer.pause();
    state.isAudioPlaying = false;
  }
  state.audioQueue = [];
  state.currentlyPlayingIndex = -1;

  const currentParent = parentId ? document.getElementById(parentId) : (document.querySelector('.werewolf-parent') || document.body);
  if (currentParent) {
    // Clear any "now-playing" highlights
    const nowPlayingElement = currentParent.querySelector('#chat-log .now-playing');
    if (nowPlayingElement) {
      nowPlayingElement.classList.remove('now-playing');
    }
  }
}
