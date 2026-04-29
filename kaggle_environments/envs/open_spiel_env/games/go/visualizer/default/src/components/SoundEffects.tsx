import { useEffect, useRef } from 'react';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import useAudio from '../hooks/useAudio';
import placeSoundUrl from '../assets/audio/go-stone-placing.mp3';
import captureSoundUrl from '../assets/audio/go-stone-removal.mp3';

const THROTTLE_MS = 150;

export default function SoundEffects() {
  const game = useGameStore((state) => state.game);
  const soundEnabled = usePreferences((state) => state.soundEnabled);
  const sounds = useAudio({ place: placeSoundUrl, capture: captureSoundUrl });
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

    const now = performance.now();
    if (now - lastPlayedRef.current < THROTTLE_MS) return;
    lastPlayedRef.current = now;

    if (placed) sounds.place.play();
    if (captured) sounds.capture.play();
  }, [game, soundEnabled, sounds]);

  return null;
}
