import { WerewolfEvent, WerewolfPlayer, WerewolfProcessedReplay, WerewolfStep } from './werewolfReplayTypes';
import { createNameReplacer, disambiguateDisplayNames } from './nameReplacer';
import { BaseGameStep, InterestingEvent, ReplayMode } from '../../types';

// Re-export for external use
export { createNameReplacer, createPlayerCapsule, disambiguateDisplayNames } from './nameReplacer';
export type { PlayerConfig, OutputFormat } from './nameReplacer';

/**
 * Get a short label for a werewolf step (e.g., the actor's action)
 * Returns format like "GPT-4 votes to exile Claude" using display names (model names)
 */
export function getWerewolfStepLabel(step: WerewolfStep): string {
  // Use the active player's display name and action text (set during transformation)
  const activePlayer = step?.players?.find((p) => p.isTurn);
  if (activePlayer?.actionDisplayText) {
    // Combine actor's display name with their action
    // activePlayer.name is already the display name (model name like "GPT-4")
    return `${activePlayer.name} ${activePlayer.actionDisplayText}`;
  }

  // Fallback for steps without an active player (e.g., game end, phase dividers)
  const event = step?.visualizerEvent;
  if (!event) return '';

  const dataType = event.dataType;

  switch (dataType) {
    case 'GameEndResultsDataEntry':
      return event.data?.winner_team ? `${event.data.winner_team} wins!` : 'Game Over';
    default:
      return event.description || event.event_name || '';
  }
}

/**
 * Get the reasoning/thoughts for a werewolf step
 */
export function getWerewolfStepDescription(step: WerewolfStep): string {
  // First check if we have thoughts from the player (set during transformation)
  const activePlayer = step?.players?.find((p) => p.isTurn);
  if (activePlayer?.thoughts) {
    return activePlayer.thoughts;
  }

  // Fallback to extracting from visualizerEvent
  const event = step?.visualizerEvent;
  if (!event) return '';

  const data = event.data;

  // The reasoning is the agent's internal thoughts
  if (data?.reasoning) {
    return data.reasoning;
  }

  // For chat entries, the message itself can serve as description
  if (event.dataType === 'ChatDataEntry' && data?.message) {
    return data.message;
  }

  // Fallback to event description
  return event.description || '';
}

/**
 * Generate action display text from an event
 * @param event The werewolf event
 * @param replaceNames Function to replace character names with display names
 */
function getActionDisplayText(event: WerewolfEvent, replaceNames: (text: string) => string = (t) => t): string {
  const data = event.data;
  const dataType = event.dataType;

  // Get target name with replacement applied
  const targetName = data?.target_id ? replaceNames(data.target_id) : undefined;

  switch (dataType) {
    case 'ChatDataEntry':
      // Also replace names in chat messages
      return data?.message ? replaceNames(data.message) : 'speaks';
    case 'DayExileVoteDataEntry':
      return targetName ? `votes to exile ${targetName}` : 'votes';
    case 'WerewolfNightVoteDataEntry':
      return targetName ? `votes to eliminate ${targetName}` : 'votes';
    case 'DoctorHealActionDataEntry':
      return targetName ? `heals ${targetName}` : 'heals';
    case 'SeerInspectActionDataEntry':
      return targetName ? `inspects ${targetName}` : 'inspects';
    case 'DayExileElectedDataEntry':
      return 'was exiled';
    case 'WerewolfNightEliminationDataEntry':
      return 'was eliminated';
    case 'SeerInspectResultDataEntry':
      return targetName ? `learned ${targetName}'s role` : 'learned a role';
    case 'DoctorSaveDataEntry':
      return 'was saved by the doctor';
    case 'GameEndResultsDataEntry':
      return `${data?.winner_team} wins!`;
    default:
      // Replace names in default description too
      const desc = event.description || '';
      return replaceNames(desc);
  }
}

const systemEntryTypeSet = new Set([
  'moderator_announcement',
  'elimination',
  'vote_request',
  'heal_request',
  'heal_result',
  'inspect_request',
  'inspect_result',
  'bidding_info',
  'bid_result',
  'day_start',
  'night_start',
]);

const visibleEventDataTypes = new Set([
  'ChatDataEntry',
  'DayExileVoteDataEntry',
  'WerewolfNightVoteDataEntry',
  'DoctorHealActionDataEntry',
  'SeerInspectActionDataEntry',
  'DayExileElectedDataEntry',
  'WerewolfNightEliminationDataEntry',
  'SeerInspectResultDataEntry',
  'DoctorSaveDataEntry',
  'GameEndResultsDataEntry',
  'PhaseDividerDataEntry',
  'DiscussionOrderDataEntry',
]);

/**
 * Build a map of player ID to player config (thumbnail, display_name, etc.)
 */
function buildPlayerConfigMap(replay: any): Map<string, any> {
  const configMap = new Map<string, any>();

  // First, get from configuration.agents
  if (replay.configuration?.agents) {
    replay.configuration.agents.forEach((agent: any) => {
      if (agent?.id) {
        configMap.set(agent.id, { ...agent });
      }
    });
  }

  // Override/supplement with GAME_END info (ground truth)
  if (replay.info?.GAME_END?.all_players) {
    replay.info.GAME_END.all_players.forEach((p: any) => {
      if (p.agent?.id) {
        const existing = configMap.get(p.agent.id) || {};
        configMap.set(p.agent.id, { ...existing, ...p.agent });
      }
    });
  }

  // Disambiguate duplicate display names (e.g., multiple "GPT-4" become "GPT-4 (1)", "GPT-4 (2)")
  disambiguateDisplayNames(configMap);

  return configMap;
}

/**
 * Get the actor ID from an event
 */
function getActorId(event: WerewolfEvent): string | undefined {
  const data = event.data;
  const dataType = event.dataType;

  // Most action events have actor_id
  if (data?.actor_id) {
    return data.actor_id;
  }

  // Some events have the target as the "actor" for display purposes
  switch (dataType) {
    case 'DayExileElectedDataEntry':
      return data?.elected_player_id;
    case 'WerewolfNightEliminationDataEntry':
      return data?.eliminated_player_id;
    case 'DoctorSaveDataEntry':
      return data?.saved_player_id;
    default:
      return undefined;
  }
}

export const werewolfTransformer = (processedReplay: any): WerewolfProcessedReplay => {
  const allEvents: WerewolfEvent[] = [];
  const newSteps: WerewolfStep[] = [];
  const originalSteps = [...(processedReplay.steps as any[])];
  const processedPhaseEvents = new Set<string>();

  let currentDisplayStep = 0;
  let allEventsIndex = 0;

  const displayStepToAllEventsIndex: number[] = [];
  const allEventsIndexToDisplayStep: number[] = [];
  const eventToKaggleStep: number[] = [];

  // Build player config map for thumbnails and display names
  const playerConfigMap = buildPlayerConfigMap(processedReplay);

  // Create memoized name replacer for efficient text processing
  const replaceNames = createNameReplacer(playerConfigMap, 'text');

  ((processedReplay.info?.MODERATOR_OBSERVATION as any[]) || []).forEach((stepEvents: any, kaggleStep: number) => {
    (stepEvents || []).flat().forEach((dataEntry: any) => {
      const event = JSON.parse(dataEntry.json_str);
      const dataType = dataEntry.data_type;
      const visibleInUI = event.visible_in_ui ?? true;

      if (!visibleInUI) return;

      if (
        event.event_name === 'day_start' ||
        event.event_name === 'night_start' ||
        event.description?.includes('Voting phase begins')
      ) {
        processedPhaseEvents.clear();
      }

      const eventFingerprint = event.description;
      if (processedPhaseEvents.has(eventFingerprint)) return;
      processedPhaseEvents.add(eventFingerprint);

      const isVisibleDataType = visibleEventDataTypes.has(dataType);
      const isVisibleEntryType = systemEntryTypeSet.has(event.event_name);

      if (!isVisibleDataType && !isVisibleEntryType) return;

      event.kaggleStep = kaggleStep;
      event.dataType = dataType;

      // Replace character names with display names in the event description
      // This ensures labels and descriptions show model names consistently
      if (event.description) {
        event.description = replaceNames(event.description);
      }

      allEvents.push(event);
      eventToKaggleStep.push(kaggleStep);

      if (dataType !== 'PhaseDividerDataEntry') {
        const stepData = originalSteps[kaggleStep];

        // Get the actor for this event
        const actorId = getActorId(event);
        const actorConfig = actorId ? playerConfigMap.get(actorId) : undefined;

        // Build a players array with the active player marked with isTurn
        const players: WerewolfPlayer[] = [];

        if (actorId) {
          // Use display_name (model name like "GPT-5.2") for consistency with visualizer
          // Fall back to actorId (character name like "Alex") if display_name not available
          const displayName = actorConfig?.display_name || actorId;

          // Replace character names with model names in the reasoning text
          const rawThoughts = event.data?.reasoning || '';
          const thoughts = replaceNames(rawThoughts);

          const activePlayer: WerewolfPlayer = {
            name: displayName,
            isTurn: true,
            thumbnail: actorConfig?.thumbnail,
            thoughts,
            actionDisplayText: getActionDisplayText(event, replaceNames),
          };
          players.push(activePlayer);
        }

        // Create a proper WerewolfStep object
        const werewolfStep: WerewolfStep = {
          step: currentDisplayStep,
          players,
          visualizerEvent: event,
          originalStepData: stepData,
        };

        newSteps.push(werewolfStep);

        displayStepToAllEventsIndex.push(allEventsIndex);
        allEventsIndexToDisplayStep[allEventsIndex] = currentDisplayStep;
        currentDisplayStep++;
      }
      allEventsIndex++;
    });
  });

  processedReplay.steps = newSteps;
  processedReplay.isTransformed = true;

  processedReplay.visualizerData = {
    allEvents,
    displayStepToAllEventsIndex,
    allEventsIndexToDisplayStep,
    eventToKaggleStep,
    originalSteps,
    // Include playerConfigMap so external components can use replaceCharacterNames
    playerConfigMap: Object.fromEntries(playerConfigMap),
  };

  return processedReplay as WerewolfProcessedReplay;
};

// Custom constants for Werewolf playback speed
const WEREWOLF_STEP_DURATION = 1000;
const WEREWOLF_TIME_PER_CHUNK = 150; // 150ms per text chunk (reading speed)

export const getWerewolfStepRenderTime = (
  gameStep: BaseGameStep,
  replayMode: ReplayMode,
  speedModifier: number
): number => {
  // Example: if we're at 2x speed, we want the render time to be half as long
  const multiplier = 1 / speedModifier;

  let currentPlayer = gameStep.players?.find((p) => p.isTurn) || {
    id: -1,
    name: 'System',
    thumbnail: '',
    isTurn: false,
    thoughts: '',
  };

  // If we should be streaming reasoning, we want the total render time to
  // account for how long it takes each token to be displayed
  if (replayMode !== 'condensed') {
    if (currentPlayer.thoughts) {
      const chunks = currentPlayer.thoughts.split(' ');
      return chunks.length * WEREWOLF_TIME_PER_CHUNK * multiplier;
    }
  }

  return WEREWOLF_STEP_DURATION * multiplier;
};

/**
 * Get interesting events for werewolf episodes.
 * Currently returns a deterministic test step plus any elimination events.
 */
export const getWerewolfStepInterestingEvents = (gameSteps: WerewolfStep[]): InterestingEvent[] => {
  if (gameSteps.length === 0) {
    return [];
  }

  const interestingEvents: InterestingEvent[] = [];

  // Test implementation: pick a deterministic step (middle of episode)
  /* const testStep = Math.floor(gameSteps.length / 2);
  interestingEvents.push({
    step: testStep,
    description: `Test event at step ${testStep + 1}`,
  });

  // Also mark elimination events as interesting (if any)
  for (let i = 0; i < gameSteps.length; i++) {
    const step = gameSteps[i];
    const event = step.visualizerEvent;
    if (event?.event_name === 'player_eliminated' || event?.event_name === 'vote_result') {
      interestingEvents.push({
        step: i,
        description: event.description || `${event.event_name} at step ${i + 1}`,
      });
    }
  } */

  return interestingEvents;
};
