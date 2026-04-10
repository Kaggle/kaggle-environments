import { useEffect, useRef } from 'react';
import { Howl } from 'howler';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import moveSoundUrl from '../assets/audio/chess-move.mp3';
import captureSoundUrl from '../assets/audio/chess-capture.mp3';

const placeSound = new Howl({ src: [moveSoundUrl], volume: 0.5 });
const captureSound = new Howl({ src: [captureSoundUrl], volume: 0.5 });

const THROTTLE_MS = 150;

export function SoundEffects() {
  const { game, options } = useGameStore();
  const soundEnabled = usePreferences((state) => state.soundEnabled);
  const lastPlayedRef = useRef(0);
  const lastStep = useRef(0);

  useEffect(() => {
    const history = game.history({ verbose: true });
    const currentMove = history.at(-1);
    const captured = currentMove?.isCapture();

    const isForwardStep = options.step > lastStep.current;
    lastStep.current = options.step;

    if (!isForwardStep) return;

    // Prevent audio spam when scrubbing quickly (e.g. holding arrow keys)
    const now = performance.now();
    if (now - lastPlayedRef.current < THROTTLE_MS) return;
    lastPlayedRef.current = now;

    placeSound.play();
    if (captured) captureSound.play();
  }, [game, options.step, soundEnabled]);

  return null;
}
