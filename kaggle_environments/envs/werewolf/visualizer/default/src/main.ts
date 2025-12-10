import { createReplayVisualizer, LegacyAdapter, processEpisodeData } from '@kaggle-environments/core';
import { renderer as legacyRenderer } from './legacy-renderer.js';
import './style.css';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

const adapter = new LegacyAdapter(legacyRenderer);

const systemEntryTypeSet = new Set([
  'moderator_announcement', 'elimination', 'vote_request', 'heal_request',
  'heal_result', 'inspect_request', 'inspect_result', 'bidding_info',
  'bid_result', 'day_start', 'night_start',
]);

const visibleEventDataTypes = new Set([
  'ChatDataEntry', 'DayExileVoteDataEntry', 'WerewolfNightVoteDataEntry',
  'DoctorHealActionDataEntry', 'SeerInspectActionDataEntry', 'DayExileElectedDataEntry',
  'WerewolfNightEliminationDataEntry', 'SeerInspectResultDataEntry', 'DoctorSaveDataEntry',
  'GameEndResultsDataEntry', 'PhaseDividerDataEntry', 'DiscussionOrderDataEntry',
]);

if (app) {
  // Set up an HMR boundary for development
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(app, adapter, {
    transformer: (replay) => {
      const processedReplay = processEpisodeData(replay, 'werewolf');
      
      const allEvents: any[] = [];
      const newSteps: any[] = [];
      const originalSteps = [...processedReplay.steps]; // Shallow copy of the steps array
      const processedPhaseEvents = new Set();
      
      let currentDisplayStep = 0;
      let allEventsIndex = 0;
      
      const displayStepToAllEventsIndex: number[] = [];
      const allEventsIndexToDisplayStep: number[] = [];
      const eventToKaggleStep: number[] = [];

      (processedReplay.info?.MODERATOR_OBSERVATION || []).forEach((stepEvents: any, kaggleStep: number) => {
        (stepEvents || []).flat().forEach((dataEntry: any) => {
            const event = JSON.parse(dataEntry.json_str);
            const dataType = dataEntry.data_type;
            const visibleInUI = event.visible_in_ui ?? true;

            if (!visibleInUI) return;

            if (event.event_name === 'day_start' || event.event_name === 'night_start' || event.description?.includes('Voting phase begins')) {
                processedPhaseEvents.clear();
            }

            let eventFingerprint = event.description;
            if (processedPhaseEvents.has(eventFingerprint)) return;
            processedPhaseEvents.add(eventFingerprint);

            const isVisibleDataType = visibleEventDataTypes.has(dataType);
            const isVisibleEntryType = systemEntryTypeSet.has(event.event_name);

            if (!isVisibleDataType && !isVisibleEntryType) return;

            event.kaggleStep = kaggleStep;
            event.dataType = dataType;
            
            allEvents.push(event);
            eventToKaggleStep.push(kaggleStep);

            if (dataType !== 'PhaseDividerDataEntry') {
                // Attach event to a CLONE of the step array
                const stepData = originalSteps[kaggleStep];
                // Clone the array to avoid mutating the shared original step
                // and to allow different properties on different display steps sharing the same kaggle step
                const stepDataClone = [...(stepData as any)]; 
                (stepDataClone as any).visualizerEvent = event;
                newSteps.push(stepDataClone);
                
                displayStepToAllEventsIndex.push(allEventsIndex);
                allEventsIndexToDisplayStep[allEventsIndex] = currentDisplayStep;
                currentDisplayStep++;
            }
            allEventsIndex++;
        });
      });

      // Update the replay object with expanded steps
      processedReplay.steps = newSteps;
      
      // Attach metadata for the renderer
      (processedReplay as any).visualizerData = {
          allEvents,
          displayStepToAllEventsIndex,
          allEventsIndexToDisplayStep,
          eventToKaggleStep,
          originalSteps // Pass the original unexpanded steps for state reconstruction
      };

      return processedReplay;
    },
  });
}