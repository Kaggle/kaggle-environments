import { useEffect, useRef } from 'react';
import placeSoundUrl from '../assets/audio/go-stone-placing.mp3';
import captureSoundUrl from '../assets/audio/go-stone-removal.mp3';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';

// Web Audio API - shared across component mounts
let audioCtx: AudioContext | null = null;
let placeBuffer: AudioBuffer | null = null;
let captureBuffer: AudioBuffer | null = null;
let buffersLoading = false;

function getAudioContext(): AudioContext {
  if (!audioCtx) {
    audioCtx = new AudioContext({ latencyHint: 'interactive' });
  }
  return audioCtx;
}

async function loadBuffers() {
  if (buffersLoading || (placeBuffer && captureBuffer)) return;
  buffersLoading = true;
  try {
    const ctx = getAudioContext();
    const [placeData, captureData] = await Promise.all([
      fetch(placeSoundUrl).then((r) => r.arrayBuffer()),
      fetch(captureSoundUrl).then((r) => r.arrayBuffer()),
    ]);
    [placeBuffer, captureBuffer] = await Promise.all([
      ctx.decodeAudioData(placeData),
      ctx.decodeAudioData(captureData),
    ]);
  } catch {
    buffersLoading = false;
  }
}

function playBuffer(buffer: AudioBuffer) {
  const ctx = getAudioContext();
  const source = ctx.createBufferSource();
  const gain = ctx.createGain();
  gain.gain.value = 0.5;
  source.buffer = buffer;
  source.connect(gain).connect(ctx.destination);
  source.start();
}

const THROTTLE_MS = 150;

export default function SoundEffects() {
  const game = useGameStore((state) => state.game);
  const soundEnabled = usePreferences((state) => state.soundEnabled);
  const prevRef = useRef({ move: 0, captures: 0 });
  const lastPlayedRef = useRef(0);

  // Resume a suspended AudioContext and pre-load buffers on first user gesture
  // (Safari requires AudioContext.resume() to be called within a user gesture)
  useEffect(() => {
    const ctx = getAudioContext();

    function unlockAudio() {
      if (ctx.state === 'suspended') ctx.resume();
      loadBuffers();
      document.removeEventListener('click', unlockAudio);
      document.removeEventListener('touchstart', unlockAudio);
      document.removeEventListener('keydown', unlockAudio);
    }

    if (ctx.state === 'running') {
      loadBuffers();
      return;
    }

    document.addEventListener('click', unlockAudio);
    document.addEventListener('touchstart', unlockAudio);
    document.addEventListener('keydown', unlockAudio);

    return () => {
      document.removeEventListener('click', unlockAudio);
      document.removeEventListener('touchstart', unlockAudio);
      document.removeEventListener('keydown', unlockAudio);
    };
  }, []);

  useEffect(() => {
    const state = game.currentState();
    const move = state.moveNumber;
    const captures = state.blackStonesCaptured + state.whiteStonesCaptured;
    const prev = prevRef.current;
    const placed = move > prev.move && state.playedPoint;
    const captured = captures > prev.captures;

    prevRef.current = { move, captures };

    if (!soundEnabled || (!placed && !captured)) return;

    // Prevent audio spam when scrubbing quickly (e.g. holding arrow keys)
    const now = performance.now();
    if (now - lastPlayedRef.current < THROTTLE_MS) return;
    lastPlayedRef.current = now;

    if (placed && placeBuffer) playBuffer(placeBuffer);
    if (captured && captureBuffer) playBuffer(captureBuffer);
  }, [game, soundEnabled]);

  return null;
}
