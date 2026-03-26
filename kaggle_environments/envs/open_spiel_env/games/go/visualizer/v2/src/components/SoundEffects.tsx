import { useEffect, useRef } from 'react';
import placeSound from '../assets/audio/go-stone-placing.mp3';
import captureSound from '../assets/audio/go-stone-removal.mp3';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';

export default function SoundEffects() {
  const game = useGameStore((state) => state.game);
  const soundEnabled = usePreferences((state) => state.soundEnabled);
  const placeRef = useRef<HTMLAudioElement>(null);
  const captureRef = useRef<HTMLAudioElement>(null);
  const prevRef = useRef({ move: 0, captures: 0 });

  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    const state = game.currentState();
    const move = state.moveNumber;
    const captures = state.blackStonesCaptured + state.whiteStonesCaptured;
    const prev = prevRef.current;
    const placed = move > prev.move && state.playedPoint;
    const captured = captures > prev.captures;

    prevRef.current = { move, captures };

    if (!soundEnabled || (!placed && !captured)) return;

    clearTimeout(timerRef.current);
    // Timeout to prevent audio spamming when scrubbing quickly (e.g. holding right arrow)
    timerRef.current = setTimeout(() => {
      if (placed && placeRef.current) {
        placeRef.current.volume = 0.5;
        placeRef.current.currentTime = 0;
        placeRef.current.play();
      }
      if (captured && captureRef.current) {
        captureRef.current.volume = 0.5;
        captureRef.current.currentTime = 0;
        captureRef.current.play();
      }
    }, 150);
  }, [game, soundEnabled]);

  return (
    <>
      <audio ref={placeRef} src={placeSound} preload="auto" />
      <audio ref={captureRef} src={captureSound} preload="auto" />
    </>
  );
}
