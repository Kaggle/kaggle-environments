export default function playNextInQueue(isContinuous = true) {
  const currentParent = document.getElementById(parentId);
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
      if (context.unstable_replayerControls) {
        context.unstable_replayerControls._replayerInstance.pause();
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

    if (displayStep !== undefined && context.unstable_replayerControls) {
      console.debug(`DEBUG: [playNextInQueue] ### ADVANCING SLIDER TO ${displayStep} ###`);
      context.unstable_replayerControls.setStep(displayStep);

      // Use a short timeout to allow the DOM to update after the step change
      setTimeout(() => {
        const freshParent = document.getElementById(parentId);
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
    }, context.speed);
  }
}