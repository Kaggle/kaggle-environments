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
  if (!audioCtx || audioCtx.state === 'closed') {
    audioCtx = new AudioContext({ latencyHint: 'interactive' });
  }
  return audioCtx;
}

// Replace the current AudioContext with a fresh one and re-decode buffers.
async function resetAudioContext() {
  if (audioCtx) {
    try {
      await audioCtx.close();
    } catch (e) {
      console.warn('Failed to close AudioContext', e);
    }
  }
  audioCtx = null;
  placeBuffer = null;
  captureBuffer = null;
  buffersLoading = false;
}

/**
 * Fetches and decodes audio.
 * Note: We fetch fresh data each time because decodeAudioData
 * detaches (neuters) the ArrayBuffer.
 */
async function loadBuffers() {
  if (buffersLoading || (placeBuffer && captureBuffer)) return;

  const ctx = getAudioContext();
  buffersLoading = true;

  try {
    const [placeRes, captureRes] = await Promise.all([fetch(placeSoundUrl), fetch(captureSoundUrl)]);
    const [placeData, captureData] = await Promise.all([placeRes.arrayBuffer(), captureRes.arrayBuffer()]);

    // decodeAudioData consumes the buffer; it cannot be reused.
    const [decodedPlace, decodedCapture] = await Promise.all([
      ctx.decodeAudioData(placeData),
      ctx.decodeAudioData(captureData),
    ]);

    placeBuffer = decodedPlace;
    captureBuffer = decodedCapture;
  } catch (error) {
    console.error('Failed to load or decode game sounds:', error);
    // Reset loading state so we can try again on next interaction
    buffersLoading = false;
  }
}

function playBuffer(buffer: AudioBuffer) {
  const ctx = getAudioContext();

  // Browsers often suspend context even after interaction if it's been idle
  if (ctx.state === 'suspended') {
    ctx.resume().catch(() => {});
  }

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

  useEffect(() => {
    async function unlockAudio() {
      const ctx = getAudioContext();

      if (ctx.state === 'suspended') {
        await ctx.resume().catch(() => {});
      }

      // If context is still not running (e.g. dead on iOS background), reset and try once more
      if (ctx.state === 'closed') {
        await resetAudioContext();
        getAudioContext();
      }

      loadBuffers();
    }

    const events = ['click', 'touchstart', 'keydown'];

    for (const event of events) document.addEventListener(event, unlockAudio);

    // Initial check if context is already available
    if (getAudioContext().state === 'running') {
      loadBuffers();
    }

    return () => {
      for (const event of events) document.removeEventListener(event, unlockAudio);
    };
  }, []);

  // Effect 2: Watch game state and trigger sounds
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

    if (placed && placeBuffer) playBuffer(placeBuffer);
    if (captured && captureBuffer) playBuffer(captureBuffer);
  }, [game, soundEnabled]);

  return null;
}
