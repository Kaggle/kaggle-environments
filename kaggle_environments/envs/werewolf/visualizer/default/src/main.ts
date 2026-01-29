import { createReplayVisualizer, LegacyAdapter, processEpisodeData } from '@kaggle-environments/core';
import { renderer as legacyRenderer } from './legacy-renderer.js';
import { tryLoadAudioMap } from './audio/AudioController.js';
import './style.css';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

const adapter = new LegacyAdapter(legacyRenderer);

if (app) {
  // Set up an HMR boundary for development
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }

  const init = async () => {
    // Resolve Episode ID from URL params (prioritized for production)
    const urlParams = new URLSearchParams(window.location.search);
    const urlEpisodeId = urlParams.get('episodeId');

    // Also check for injected REPLAY data (standard for some Kaggle loaders)
    const replayData = (window as any).REPLAY;
    const jsonEpisodeId = replayData?.info?.EpisodeId || replayData?.id;

    const episodeId = urlEpisodeId || jsonEpisodeId;

    // Ensure kaggleWerewolf state exists (should be initialized by AudioController import)
    const audioState = (window as any).kaggleWerewolf || {};
    if (episodeId) {
      audioState.episodeId = episodeId;
    }

    const envUrl = import.meta.env.VITE_AUDIO_MAP_FILE;

    // Centralized discovery and loading
    await tryLoadAudioMap(episodeId, envUrl);

    createReplayVisualizer(app, adapter, {
      transformer: (replay: any) => {
        // Final fallback: if we only just received the episodeId from the replayer, trigger load
        const finalId = replay?.info?.EpisodeId || replay.id;
        if (finalId && !(window as any).AUDIO_MAP) {
          tryLoadAudioMap(finalId, envUrl);
        }
        return processEpisodeData(replay, 'werewolf');
      },
    });
  };

  init();
}
