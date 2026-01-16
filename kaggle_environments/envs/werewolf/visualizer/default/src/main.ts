import { createReplayVisualizer, LegacyAdapter, processEpisodeData } from '@kaggle-environments/core';
import { renderer as legacyRenderer } from './legacy-renderer.js';
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
    // Check if we need to load an external audio map (for dev-with-audio mode)
    const audioMapFile = import.meta.env.VITE_AUDIO_MAP_FILE;
    if (audioMapFile) {
      try {
        console.log(`Loading audio map from: ${audioMapFile}`);
        const response = await fetch(audioMapFile);
        const data = await response.json();

        // Rebase audio paths relative to the map file
        const audioMapDir = audioMapFile.substring(0, audioMapFile.lastIndexOf('/') + 1);
        if (audioMapDir) {
          for (const key in data) {
            if (typeof data[key] === 'string' && !data[key].startsWith('http') && !data[key].startsWith('/')) {
              data[key] = audioMapDir + data[key];
            }
          }
        }

        (window as any).AUDIO_MAP = data;
        console.log("Audio map loaded successfully.");
      } catch (e) {
        console.error(`Failed to load audio map from ${audioMapFile}:`, e);
      }
    }

    createReplayVisualizer(app, adapter, {
      transformer: (replay) => processEpisodeData(replay, 'werewolf'),
    });
  };

  init();
}