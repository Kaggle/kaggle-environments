import { useEffect, useRef } from 'react';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import useAudio from '../hooks/useAudio';
import moveSoundUrl from '../assets/audio/chess-move.mp3';
import captureSoundUrl from '../assets/audio/chess-capture.mp3';
import { SCRUB_THRESHOLD_MS } from '../constants';

export function SoundEffects() {
  const { game, options } = useGameStore();
  const soundEnabled = usePreferences((state) => state.soundEnabled);
  const sounds = useAudio({ move: moveSoundUrl, capture: captureSoundUrl });
  const lastPlayedRef = useRef(0);
  const lastStep = useRef(0);

  useEffect(() => {
    const isForwardStep = options.step > lastStep.current;
    lastStep.current = options.step;

    if (!isForwardStep || !soundEnabled) return;

    const history = game.history({ verbose: true });
    const currentMove = history.at(-1);
    const captured = currentMove?.isCapture();

    const now = performance.now();
    if (now - lastPlayedRef.current < SCRUB_THRESHOLD_MS) return;
    lastPlayedRef.current = now;

    sounds.move.play();
    if (captured) sounds.capture.play();
  }, [game, options.step, sounds, soundEnabled]);

  return null;
}
