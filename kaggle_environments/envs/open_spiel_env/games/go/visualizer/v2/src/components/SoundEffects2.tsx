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
    console.log('new audioContext');
  }
  return audioCtx;
}

/** Replace the current AudioContext with a fresh one and re-decode buffers. */
function resetAudioContext() {
  console.warn('reset audio context');
  audioCtx?.close().catch(() => {});
  audioCtx = null;
  placeBuffer = null;
  captureBuffer = null;
  buffersLoading = false;
}

async function loadBuffers() {
  console.log(1);
  if (buffersLoading || (placeBuffer && captureBuffer)) return;
  console.log(2);
  buffersLoading = true;
  try {
    const ctx = getAudioContext();
    console.log(3);
    const [placeData, captureData] = await Promise.all([
      fetch(placeSoundUrl).then((r) => r.arrayBuffer()),
      fetch(captureSoundUrl).then((r) => r.arrayBuffer()),
    ]);
    console.log(4);
    [placeBuffer, captureBuffer] = await Promise.all([
      ctx.decodeAudioData(placeData),
      ctx.decodeAudioData(captureData),
    ]);
    console.log(5);
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

  // Some operating systems will kill existing AudioContext instances
  // unprompted, e.g. on iOS when the browser is backgrounded.
  useEffect(() => {
    function unlockAudio() {
      let ctx = getAudioContext();
      console.warn(ctx.state);
      if (ctx.state === 'running') {
        loadBuffers();
        return;
      }

      // If the context is not running, attempt to re-create the context.
      // This has to happen synchronously for iOS to see it as a
      // user-interaction.
      resetAudioContext();
      ctx = getAudioContext();
      ctx.resume().catch(() => {});
      loadBuffers();
    }

    document.addEventListener('click', unlockAudio);
    document.addEventListener('touchstart', unlockAudio);
    document.addEventListener('keydown', unlockAudio);

    if (getAudioContext().state === 'running') loadBuffers();

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
