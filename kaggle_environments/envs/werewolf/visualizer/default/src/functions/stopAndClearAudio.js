export default function stopAndClearAudio(audioState, parentId) {
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