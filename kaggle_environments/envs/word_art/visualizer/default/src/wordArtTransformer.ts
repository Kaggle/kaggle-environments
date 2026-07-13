import { BaseGameStep, ReplayData } from '@kaggle-environments/core';

export interface WordArtStep extends BaseGameStep {
  rawAgents: any[];
}

const TEAM_LABEL = ['Blue', 'Blue', 'Yellow', 'Yellow'] as const;

/**
 * Populate the side panel with per-step player actions + LLM thoughts.
 *
 * Convention (matches the Kaggle simultaneous-move schema):
 *   ``steps[i][j].action`` is what agent j SUBMITTED at step i (based on
 *   the observation at step i-1). So "was this agent active on this
 *   step" is decided by ``steps[i-1][j].status``, and the role/target
 *   they saw when producing the action lives on ``steps[i-1][j].observation``.
 *
 * We surface: role-aware action text (drew art / guessed a word), and
 * the raw ``thoughts`` field written by ``core_harness`` for LLM agents.
 * The side panel's default label/description getters read these off
 * ``player.actionDisplayText`` and ``player.thoughts``.
 *
 * Splitting simultaneous moves. The sidebar only surfaces the first
 * player with ``isTurn=true`` per step, so when both teams move at the
 * same tick (both artists submitting, or both guessers guessing on the
 * same attempt) one team's action and thoughts would be dropped from
 * the log. We split those ticks into one sub-step per moving agent,
 * ordered by agent index (blue before yellow). All sub-steps share the
 * same ``rawAgents`` reference so the renderer draws identical game
 * state across them — only which player's card is highlighted changes.
 */
export const wordArtTransformer = (environment: ReplayData, _gameName: string): ReplayData => {
  const rawSteps = environment.steps as unknown as any[][];
  const transformedSteps: WordArtStep[] = [];

  rawSteps.forEach((stepAgents, index) => {
    const prevStep = index > 0 ? rawSteps[index - 1] : null;

    const allPlayers = stepAgents.map((agent: any, idx: number) => {
      const prevAgent = prevStep?.[idx];
      const prevObs = prevAgent?.observation ?? {};
      const isTurn = prevAgent?.status === 'ACTIVE';
      const role: string = prevObs.role ?? '';

      // core_harness wraps as {submission, thoughts, status, ...}.
      const rawAction = agent.action;
      const isCoreHarness = typeof rawAction === 'object' && rawAction !== null && 'submission' in rawAction;
      const submission = isCoreHarness ? rawAction.submission : rawAction;
      const thoughts: string = (isCoreHarness && rawAction?.thoughts) || '';

      let actionDisplayText = '';
      if (isTurn) {
        if (role === 'artist') {
          const artStr = typeof submission === 'string' ? submission : '';
          if (!artStr) {
            actionDisplayText = 'Drew (empty submission)';
          } else {
            // Intentionally NOT including a preview of the art itself:
            // the ASCII drawing needs a monospace font and multi-line
            // layout to be readable, so a one-line preview in the log
            // is just noise. The main renderer shows the actual art.
            const target = prevObs.target_word ?? '';
            actionDisplayText = target
              ? `Drew for "${target}" — ${artStr.length} chars`
              : `Drew ${artStr.length} chars`;
          }
        } else if (role === 'guesser') {
          const guess = typeof submission === 'string' ? submission : String(submission ?? '');
          actionDisplayText = guess ? `Guessed: ${guess}` : 'Guessed (empty)';
        }
      }

      const teamNames = environment.info?.TeamNames ?? [];
      const teamName = teamNames[idx];
      const roleTag = role ? ` — ${role}` : '';
      const baseName = teamName || `${TEAM_LABEL[idx]} Player ${(idx % 2) + 1}`;
      const displayName = `${baseName} (${TEAM_LABEL[idx]}${roleTag})`;

      return {
        id: idx,
        name: displayName,
        thumbnail: '',
        isTurn,
        actionDisplayText,
        thoughts,
      };
    });

    const activeIndices = allPlayers.filter((p) => p.isTurn).map((p) => p.id);
    if (activeIndices.length <= 1) {
      transformedSteps.push({ step: transformedSteps.length, players: allPlayers, rawAgents: stepAgents });
      return;
    }

    // Multiple simultaneous moves: emit one sub-step per moving agent
    // (blue-first by agent index). Each sub-step highlights exactly one
    // player so the sidebar's `find(p => p.isTurn)` picks the right one.
    for (const activeId of activeIndices) {
      const players = allPlayers.map((p) => ({ ...p, isTurn: p.id === activeId }));
      transformedSteps.push({ step: transformedSteps.length, players, rawAgents: stepAgents });
    }
  });

  return { ...environment, steps: transformedSteps };
};
