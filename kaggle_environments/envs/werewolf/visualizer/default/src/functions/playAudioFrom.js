export default function playAudioFrom(startIndex, isContinuous = true) {
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