import { CABTReplay, CABTStep, CABTPlayer, VisualizeStep } from './types';

function deriveWinner(result: number, teamNames: string[]): string | null {
  if (result === -1) {
    return null;
  }
  if (result === 2) {
    return 'Draw';
  }
  if (teamNames[result]) {
    return `🎉 ${teamNames[result]} Wins!`;
  }
  return `Player ${result} Wins!`;
}

export const cabtTransformer = (environment: any): CABTStep[] => {
  const cabtReplay = environment as CABTReplay;
  const teamNames = cabtReplay.info.TeamNames;

  // The visualization data is what we need to render. It's nested weirdly.
  const visualizeSteps: VisualizeStep[] = cabtReplay.steps?.[0]?.[0]?.visualize || [];

  if (visualizeSteps.length === 0) {
    return [];
  }

  const cabtSteps: CABTStep[] = visualizeSteps.map((visStep, index) => {
    const currentState = visStep.current;

    const players: CABTPlayer[] = currentState.players.map((rawPlayer, playerIndex) => {
      return {
        id: playerIndex,
        name: teamNames[playerIndex] || `Player ${playerIndex}`,
        isTurn: currentState.yourIndex === playerIndex,
        thumbnail: '',

        active: rawPlayer.active,
        bench: rawPlayer.bench,
        hand: rawPlayer.hand,
        deckCount: rawPlayer.deckCount,
        discardCount: rawPlayer.discard.length,
        prizeCount: rawPlayer.prize.length,

        asleep: rawPlayer.asleep,
        burned: rawPlayer.burned,
        confused: rawPlayer.confused,
        paralyzed: rawPlayer.paralyzed,
        poisoned: rawPlayer.poisoned,
      };
    });

    const isTerminal = currentState.result !== -1;
    const winner = deriveWinner(currentState.result, teamNames);

    const step: CABTStep = {
      step: index,
      players,
      stadium: currentState.stadium,
      result: currentState.result,
      isTerminal,
      winner,
      turn: currentState.turn,
    };

    return step;
  });

  // Add a final step to show the result clearly if the game ended
  const lastVisStep = visualizeSteps[visualizeSteps.length - 1];
  if (lastVisStep && lastVisStep.current.result !== -1) {
    const lastStep = cabtSteps[cabtSteps.length - 1];
    if (lastStep) {
      cabtSteps.push({
        ...lastStep,
        step: cabtSteps.length,
        isTerminal: true,
        players: lastStep.players.map((p) => ({ ...p, isTurn: false })),
      });
    }
  }

  return cabtSteps;
};
