import { create } from 'zustand';
import { Game } from 'tenuki';

interface GoStore {
  go: Game;
  setState: (go: Game) => void;
}

const useGoStore = create<GoStore>((set) => ({
  go: new Game({ boardSize: 9 }),
  setState: (go: Game) => set({ go }),
}));

export default useGoStore;
