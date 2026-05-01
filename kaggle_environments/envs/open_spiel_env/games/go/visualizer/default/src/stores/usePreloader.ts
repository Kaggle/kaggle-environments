import { create } from 'zustand';

interface PreloaderStore {
  pixiReady: boolean;
  assetsReady: boolean;
  setPixiReady: () => void;
  setAssetsReady: () => void;
}

const usePreloader = create<PreloaderStore>((set) => ({
  pixiReady: false,
  assetsReady: false,
  setPixiReady: () => set({ pixiReady: true }),
  setAssetsReady: () => set({ assetsReady: true }),
}));

export default usePreloader;
