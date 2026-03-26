import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Preferences {
  showTerritory: boolean;
  showHeroAnimations: boolean;
  showAnnotations: boolean;
  soundEnabled: boolean;
  reducedMotion: boolean;
  toggle: (key: 'showTerritory' | 'showHeroAnimations' | 'showAnnotations' | 'soundEnabled' | 'reducedMotion') => void;
}

const usePreferences = create<Preferences>()(
  persist(
    (set) => ({
      showTerritory: true,
      showHeroAnimations: true,
      showAnnotations: true,
      soundEnabled: false,
      reducedMotion: false,
      toggle: (key) => set((state) => ({ [key]: !state[key] })),
    }),
    {
      name: 'go-visualizer-preferences',
      partialize: ({ soundEnabled, toggle, ...rest }) => rest,
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
