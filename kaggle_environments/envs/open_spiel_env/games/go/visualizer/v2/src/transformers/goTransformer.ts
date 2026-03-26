/* eslint-disable @typescript-eslint/no-explicit-any */
import { GoPlayer, GoReplay, GoStep, GoBoardState, GoReplayStep } from './goReplayTypes';

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
      board_size: 13,
      komi: 7.5,
      current_player_to_move: '',
      move_number: 0,
      previous_move_a1: null,
      board: [],
    };
  }
}

function deriveWinner(step: GoReplayStep[]): string | null {
  if (step[0].observation.isTerminal === false) return null;
  if (step[0].reward === step[1].reward) return 'Draw';
  return step[0].reward === 1 ? 'Black Wins!' : 'White Wins!';
}

export const goTransformer = (environment: any): GoStep[] => {
  const goReplay = environment as GoReplay;
  const goSteps: GoStep[] = [];

  const firstStep = goReplay.steps[0];
  const extraStepPlayers = [0, 1].map(
    (index): GoPlayer => ({
      id: index,
      name: environment.info.TeamNames[index],
      thumbnail: '',
      isTurn: false,
      actionDisplayText: '',
      thoughts: '',
      reward: null,
      generateReturns: null,
    })
  );

  goSteps.push({
    step: goSteps.length,
    players: extraStepPlayers,
    boardState: parseBoardState(firstStep[0].observation.observationString),
    isTerminal: false,
    winner: null,
  });

  for (const step of goReplay.steps) {
    if (step.some((p) => p.action?.actionString) === false) continue;

    const stepPlayers: GoPlayer[] = step.map(
      (player, index): GoPlayer => ({
        id: index,
        name: environment.info.TeamNames[index],
        thumbnail: '',
        isTurn: player.action?.submission !== undefined && player.action.submission !== -1,
        actionDisplayText: player.action?.actionString?.split(' ').at(1) ?? '',
        thoughts: parseThoughts(player.action),
        reward: player.reward,
        generateReturns: player.action?.generate_returns ?? null,
      })
    );

    goSteps.push({
      step: goSteps.length,
      players: stepPlayers,
      boardState: parseBoardState(step[0].observation.observationString),
      isTerminal: step[0].observation.isTerminal,
      winner: null,
    });
  }

  const lastReplayStep = goReplay.steps[goReplay.steps.length - 1];

  goSteps.push({
    step: goSteps.length,
    players: goSteps[goSteps.length - 1].players,
    boardState: goSteps[goSteps.length - 1].boardState,
    isTerminal: lastReplayStep[0].observation.isTerminal,
    winner: deriveWinner(lastReplayStep),
  });

  return goSteps;
};
