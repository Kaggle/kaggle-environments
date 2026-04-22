import { BaseGameStep, ReplayData } from '@kaggle-environments/core';

export interface WordAssociationStep extends BaseGameStep {
  rawAgents: any[];
}

export const wordAssociationTransformer = (environment: ReplayData, _gameName: string): ReplayData => {
  const rawSteps = environment.steps as unknown as any[][];
  const transformedSteps: WordAssociationStep[] = [];

  rawSteps.forEach((stepAgents, index) => {
    // The previous step determines whose turn it is this step.
    const prevTurnData = index > 0 ? rawSteps[index - 1] : null;
    const currentTurnVal = prevTurnData ? prevTurnData[0]?.observation?.current_turn : -1;

    const players = stepAgents.map((agent: any, idx: number) => {
      const action = agent.action;

      let thoughts = '';
      let actionDisplayText = '';
      let isTurn = currentTurnVal === idx;

      if (typeof action === 'object' && action !== null) {
        if ('thinking' in action) {
          thoughts = action.thinking;
        }
        if ('clue' in action) {
          actionDisplayText = `Clue: ${action.clue} (${action.number})`;
        } else if ('guess' in action) {
          const word = action.guess === -1 ? null : prevTurnData?.[0]?.observation?.words?.[action.guess];
          actionDisplayText = action.guess === -1 ? 'Passed' : `Guessed: ${word || action.guess}`;
        }
      } else if (typeof action === 'number') {
        // Fallback for non-LLM agents using numbers
        const word = action === -1 ? null : prevTurnData?.[0]?.observation?.words?.[action];
        actionDisplayText = action === -1 ? 'Passed' : `Guessed: ${word || action}`;
      }

      // If they passed but were not expected to move, clear the text
      if (!isTurn && action === 0) {
        actionDisplayText = '';
      }

      const roleNames = ['Blue Spymaster', 'Blue Guesser', 'Yellow Spymaster', 'Yellow Guesser'];
      const baseName = roleNames[idx] || `Agent ${idx}`;
      const teamName = environment.info?.TeamNames?.[idx];
      const displayName = teamName ? `${baseName} (${teamName})` : baseName;

      return {
        id: idx,
        name: displayName,
        thumbnail: '',
        isTurn: isTurn,
        actionDisplayText,
        thoughts,
      };
    });

    transformedSteps.push({
      step: index,
      players,
      rawAgents: stepAgents,
    });
  });

  return {
    ...environment,
    steps: transformedSteps,
  };
};
