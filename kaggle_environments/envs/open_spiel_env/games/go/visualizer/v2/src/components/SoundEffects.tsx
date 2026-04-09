import { useEffect, useRef } from 'react';
import { Howl } from 'howler';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import placeSoundUrl from '../assets/audio/go-stone-placing.mp3';
import captureSoundUrl from '../assets/audio/go-stone-removal.mp3';

const placeSound = new Howl({ src: [placeSoundUrl], volume: 0.5 });
const captureSound = new Howl({ src: [captureSoundUrl], volume: 0.5 });

const THROTTLE_MS = 150;

export default function SoundEffects() {
  const game = useGameStore((state) => state.game);
  const soundEnabled = usePreferences((state) => state.soundEnabled);
  const prevRef = useRef({ move: 0, captures: 0 });
  const lastPlayedRef = useRef(0);

  useEffect(() => {
    const state = game.currentState();
    const move = state.moveNumber;
    const captures = (state.blackStonesCaptured || 0) + (state.whiteStonesCaptured || 0);
    const prev = prevRef.current;

    const placed = move > prev.move && state.playedPoint;
    const captured = captures > prev.captures;

    prevRef.current = { move, captures };

    if (!soundEnabled || (!placed && !captured)) return;

    // Prevent audio spam when scrubbing quickly (e.g. holding arrow keys)
    const now = performance.now();
    if (now - lastPlayedRef.current < THROTTLE_MS) return;
    lastPlayedRef.current = now;

    if (placed) placeSound.play();
    if (captured) captureSound.play();
  }, [game, soundEnabled]);

  return null;
}
