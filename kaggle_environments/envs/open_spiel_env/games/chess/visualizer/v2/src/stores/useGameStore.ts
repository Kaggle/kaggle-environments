import { create } from 'zustand';
import { Chess } from 'chess.js';

interface GameStore {
  game: Chess;
  setState: (game: Chess) => void;
}

const useGameStore = create<GameStore>((set) => ({
  game: new Chess(),
  setState: (game: Chess) => set({ game }),
}));

export default useGameStore;
