import { create } from 'zustand';
import { createGame, GameState } from 'jgoboard';

interface GoStore {
  go: GameState;
  setState: (go: GameState) => void;
}

const useGoStore = create<GoStore>((set) => ({
  go: createGame({ size: 9 }),
  setState: (go: GameState) => set({ go }),
}));

export default useGoStore;
