import { GoReplay, GoPlayer, GoStep, GoBoardState } from './goReplayTypes';

function parseThoughts(action?: { generate_returns?: string[]; thoughts?: string }): string {
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) {
        return parsed.main_response_and_thoughts;
      }
    } catch {
      // Fall through to use action.thoughts
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

function deriveWinnerFromRewards(players: GoPlayer[]): string {
  if (players.length < 2) return '';

  const player0Reward = players[0].reward;
  const player1Reward = players[1].reward;

  if (player0Reward === player1Reward) {
    return 'Draw';
  }

  const winnerPlayerIndex = player0Reward === 1 ? 0 : 1;

  return `${players[winnerPlayerIndex].name} Wins!`;
}

export const goTransformer = (environment: any): GoStep[] => {
  const goReplay = environment as GoReplay;

  const goSteps: GoStep[] = [];

  goReplay.steps.forEach((step, index) => {
    const stepPlayers: GoPlayer[] = step.map((player, playerIndex): GoPlayer => {
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
        reward: player.reward,
      };
    });

    if (stepPlayers.some((player) => player.isTurn)) {
      goSteps.push({
        step: index,
        players: stepPlayers,
        boardState: parseBoardState(step[0].observation.observationString),
        isTerminal: step[0].observation.isTerminal,
        winner: null,
      });
    }
  });

  if (goSteps.length > 0) {
    const lastStep = goSteps[goSteps.length - 1];
    if (lastStep.isTerminal) {
      lastStep.winner = deriveWinnerFromRewards(lastStep.players);
    }
  }

  return goSteps;
};
