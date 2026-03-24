import { memo, useEffect, useRef } from 'react';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import placeSound from '../assets/audio/go-stone-placing.mp3';
import captureSound from '../assets/audio/go-stone-removal.mp3';

export default memo(function SoundEffects() {
  const game = useGameStore((s) => s.game);
  const soundEnabled = usePreferences((s) => s.soundEnabled);
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
        placeRef.current.currentTime = 0;
        placeRef.current.play();
      }
      if (captured && captureRef.current) {
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
});
