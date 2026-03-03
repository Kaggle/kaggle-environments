import { create } from 'zustand';
import { Game } from 'tenuki';
import { GoStep, GameRendererProps } from '@kaggle-environments/core';

interface GameStore {
  game: Game;
  options: GameRendererProps<GoStep[]> | null,
  setState: (game: Game, options: GameRendererProps<GoStep[]>) => void;
}

const useGameStore = create<GameStore>((set) => ({
  game: new Game({ boardSize: 9 }),
  options: null,
  setState: (game: Game, options: GameRendererProps<GoStep[]>) => set({ game, options }),
}));

export default useGameStore;
