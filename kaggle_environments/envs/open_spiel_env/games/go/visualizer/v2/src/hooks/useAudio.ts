import { useEffect, useRef } from 'react';
import { Howl, Howler } from 'howler';

type SoundConfig = { src: string; volume?: number };
type SoundMap<T extends string> = Record<T, SoundConfig | string>;
type Sounds<T extends string> = Record<T, Howl>;

function createSounds<T extends string>(map: SoundMap<T>): Sounds<T> {
  const entries = Object.entries<SoundConfig | string>(map).map(([key, value]) => {
    const config = typeof value === 'string' ? { src: value } : value;
    return [key, new Howl({ src: [config.src], volume: config.volume ?? 0.5 })];
  });
  return Object.fromEntries(entries) as Sounds<T>;
}

export default function useAudio<T extends string>(map: SoundMap<T>): Sounds<T> {
  const soundsRef = useRef<Sounds<T>>(createSounds(map));

  useEffect(() => {
    function cleanup() {
      window.removeEventListener('pointerdown', resume);
      window.removeEventListener('keydown', resume);
    }

    function resume() {
      Howler.unload();
      soundsRef.current = createSounds(map);
      cleanup();
    }

    const handleVisibilityChange = () => {
      if (document.visibilityState !== 'visible') return;
      const state = Howler.ctx?.state as string;
      if (state !== 'interrupted' && state !== 'suspended') return;

      window.addEventListener('pointerdown', resume);
      window.addEventListener('keydown', resume);
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      cleanup();
    };
  }, []);

  return soundsRef.current;
}
