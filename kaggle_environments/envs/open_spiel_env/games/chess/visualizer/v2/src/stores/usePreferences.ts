import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface PreferencesState {
  showHeroAnimations: boolean;
  showAnnotations: boolean;
  showHighlights: boolean;
  soundEnabled: boolean;
  reducedMotion: boolean;
}

interface Preferences extends PreferencesState {
  toggle: (key: keyof PreferencesState) => void;
}

const usePreferences = create<Preferences>()(
  persist(
    (set) => ({
      showHeroAnimations: true,
      showAnnotations: true,
      showHighlights: true,
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
        showHighlights: state.showHighlights,
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
