import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Preferences {
  showHeroAnimations: boolean;
  showAnnotations: boolean;
  soundEnabled: boolean;
  reducedMotion: boolean;
  toggle: (key: 'showHeroAnimations' | 'showAnnotations' | 'soundEnabled' | 'reducedMotion') => void;
}

const usePreferences = create<Preferences>()(
  persist(
    (set) => ({
      showHeroAnimations: true,
      showAnnotations: true,
      soundEnabled: false,
      reducedMotion: false,
      toggle: (key) => set((state) => ({ [key]: !state[key] })),
    }),
    {
      name: 'go-visualizer-preferences',
      // Explicitly define which keys should persist across sessions.
      partialize: (state) => ({
        showHeroAnimations: state.showHeroAnimations,
        showAnnotations: state.showAnnotations,
        reducedMotion: state.reducedMotion,
      }),
    }
  )
);

// Sync reducedMotion to a data attribute on <html> for CSS to consume
const syncReducedMotion = (state: Preferences) => {
  document.documentElement.toggleAttribute('data-reduced-motion', state.reducedMotion);
};
syncReducedMotion(usePreferences.getState());
usePreferences.subscribe(syncReducedMotion);

export default usePreferences;
