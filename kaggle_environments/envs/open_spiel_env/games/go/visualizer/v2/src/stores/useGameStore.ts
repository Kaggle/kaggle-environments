import { create } from 'zustand';
import { createGame, GameState } from 'jgoboard';

interface GameStore {
  game: GameState;
  setState: (go: GameState) => void;
}

const useGameStore = create<GameStore>((set) => ({
  game: createGame({ size: 9 }),
  setState: (game: GameState) => set({ game }),
}));

export default useGameStore;
