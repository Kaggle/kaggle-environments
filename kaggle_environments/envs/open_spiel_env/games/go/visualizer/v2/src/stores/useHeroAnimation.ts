import { create } from 'zustand';

interface HeroAnimationState {
  src: string | null;
  onClose: (() => void) | undefined;
  play: (src: string, onClose?: () => void) => void;
  close: () => void;
  cancel: () => void;
}

const useHeroAnimation = create<HeroAnimationState>((set, get) => ({
  src: null,
  onClose: undefined,
  play: (src, onClose) => set({ src, onClose }),
  close: () => {
    const { onClose } = get();
    set({ src: null, onClose: undefined });
    onClose?.();
  },
  cancel: () => set({ src: null, onClose: undefined }),
}));

export default useHeroAnimation;
