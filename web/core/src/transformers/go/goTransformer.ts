import { GoReplay, GoStep, GoBoardState, GoReplayStep } from './goReplayTypes';
import { BaseGamePlayer } from '../../types';

function parseThoughts(action?: { generate_returns?: string[]; thoughts?: string }): string {
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      return action?.thoughts ?? '';
    }
  }
  return action?.thoughts ?? '';
}

function parseBoardState(observationString: string): GoBoardState {
  try {
    const obs = JSON.parse(observationString);
    const board: string[][] = obs.board_grid.map((row: Record<string, string>[]) =>
      row.map((cell) => Object.values(cell)[0])
    );

    return {
      board_size: obs.board_size,
      komi: obs.komi,
      current_player_to_move: obs.current_player_to_move,
      move_number: obs.move_number,
      previous_move_a1: obs.previous_move_a1,
      board,
    };
  } catch {
    return {
      board_size: 9,
      komi: 7.5,
      current_player_to_move: '',
      move_number: 0,
      previous_move_a1: null,
      board: [],
    };
  }
}

function deriveWinner(step: GoReplayStep[]): string {
  if (step.length < 2) return '';

  const reward0 = step[0].reward;
  const reward1 = step[1].reward;

  if (reward0 === reward1) {
    return 'Draw';
  }

  return reward0 === 1 ? 'Black Wins!' : 'White Wins!';
}

export const goTransformer = (environment: any): GoStep[] => {
  const goReplay = environment as GoReplay;

  const goSteps: GoStep[] = [];

  goReplay.steps.forEach((step, index) => {
    const stepPlayers: BaseGamePlayer[] = step.map((player, playerIndex) => {
      const actionString = player.action?.actionString ?? '';
      const [colorCode, move] = actionString.split(' ');
      const colorName = colorCode === 'W' ? 'White' : 'Black';

      return {
        id: playerIndex,
        name: colorName,
        thumbnail: '',
        isTurn: player.action?.submission !== undefined && player.action.submission !== -1,
        actionDisplayText: move ?? '',
        thoughts: parseThoughts(player.action),
      };
    });

    if (stepPlayers.some((player) => player.isTurn)) {
      const isTerminal = step[0].observation.isTerminal;
      goSteps.push({
        step: index,
        players: stepPlayers,
        boardState: parseBoardState(step[0].observation.observationString),
        isTerminal,
        winner: isTerminal ? deriveWinner(step) : null,
      });
    }
  });

  return goSteps;
};
