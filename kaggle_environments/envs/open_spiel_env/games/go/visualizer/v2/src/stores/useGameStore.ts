import { create } from 'zustand';
import { Game } from 'tenuki';

interface GameStore {
  game: Game;
  setState: (go: Game) => void;
}

const useGameStore = create<GameStore>((set) => ({
  game: new Game({ boardSize: 9 }),
  setState: (game: Game) => set({ game }),
}));

export default useGameStore;
