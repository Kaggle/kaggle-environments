export default function loadQueueFrom(startIndex) {
  console.debug(`DEBUG: [loadQueueFrom] Loading queue from index: ${startIndex}`);
  if (!window.werewolfGamePlayer || !window.werewolfGamePlayer.allEvents) {
      console.error('DEBUG: [loadQueueFrom] CRITICAL: allEvents not found.');
      return;
  }
  const allEvents = window.werewolfGamePlayer.allEvents;
  const eventsToPlay = allEvents.slice(startIndex);
  console.debug(`DEBUG: [loadQueueFrom] Found ${eventsToPlay.length} potential events.`);

  audioState.audioQueue = []; // Clear previous queue

  if (eventsToPlay.length > 0) {
      eventsToPlay.forEach((entry, i) => {
      const allEventsIndex = startIndex + i;

      let audioEventDetails = null;
      const data = entry.data || {};
      const event_name = entry.event_name;
      const description = entry.description || '';
      const day_count = entry.day;

      // This logic is to identify if an event should have audio
      // and what the audio content is.
      switch (entry.dataType) {
          case 'ChatDataEntry':
          if (data.actor_id && data.actor_id !== 'moderator' && data.message) {
              audioEventDetails = { message: data.message, speaker: data.actor_id };
          }
          break;
          case 'DayExileVoteDataEntry':
          if (data.actor_id && data.target_id) {
              audioEventDetails = {
              message: `${data.actor_id} votes to exile ${data.target_id}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'WerewolfNightVoteDataEntry':
          if (data.actor_id && data.target_id) {
              audioEventDetails = {
              message: `${data.actor_id} votes to eliminate ${data.target_id}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'SeerInspectActionDataEntry':
          if (data.actor_id && data.target_id) {
              audioEventDetails = { message: `${data.actor_id} inspects ${data.target_id}.`, speaker: 'moderator' };
          }
          break;
          case 'DoctorHealActionDataEntry':
          if (data.actor_id && data.target_id) {
              audioEventDetails = { message: `${data.actor_id} heals ${data.target_id}.`, speaker: 'moderator' };
          }
          break;
          case 'DayExileElectedDataEntry':
          if (data.elected_player_id && data.elected_player_role_name) {
              audioEventDetails = {
              message: `${data.elected_player_id} was exiled by vote. Their role was a ${data.elected_player_role_name}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'WerewolfNightEliminationDataEntry':
          if (data.eliminated_player_id && data.eliminated_player_role_name) {
              audioEventDetails = {
              message: `${data.eliminated_player_id} was eliminated. Their role was a ${data.eliminated_player_role_name}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'DoctorSaveDataEntry':
          if (data.saved_player_id) {
              audioEventDetails = {
              message: `${data.saved_player_id} was attacked but saved by a Doctor!`,
              speaker: 'moderator',
              };
          }
          break;
          case 'GameEndResultsDataEntry':
          if (data.winner_team) {
              audioEventDetails = {
              message: `The game is over. The ${data.winner_team} team has won!`,
              speaker: 'moderator',
              };
          }
          break;
          case 'WerewolfNightEliminationElectedDataEntry':
          if (data.elected_target_player_id) {
              audioEventDetails = {
              message: `The werewolves have chosen to eliminate ${data.elected_target_player_id}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'SeerInspectResultDataEntry':
          if (data.role) {
              audioEventDetails = {
              message: `${data.actor_id} saw ${data.target_id}'s role is ${data.role}.`,
              speaker: 'moderator',
              };
          } else if (data.team) {
              audioEventDetails = {
              message: `${data.actor_id} saw ${data.target_id}'s team is ${data.team}.`,
              speaker: 'moderator',
              };
          }
          break;
          case 'DiscussionOrderDataEntry':
          audioEventDetails = { message: description, speaker: 'moderator' };
      }

      if (!audioEventDetails && event_name === 'moderator_announcement') {
          if (description.includes('discussion rule is')) {
          audioEventDetails = { message: 'Discussion begins!', speaker: 'moderator' };
          } else if (description.includes('Voting phase begins')) {
          audioEventDetails = { message: 'Exile voting begins!', speaker: 'moderator' };
          } else {
          audioEventDetails = { message: entry.description, speaker: 'moderator' };
          }
      } else if (!audioEventDetails && event_name === 'day_start') {
          audioEventDetails = { message: `Day ${day_count} begins!`, speaker: 'moderator' };
      } else if (!audioEventDetails && event_name === 'night_start') {
          audioEventDetails = { message: `Night ${day_count} begins!`, speaker: 'moderator' };
      }

      // Every event goes into the queue.
      audioState.audioQueue.push({
          allEventsIndex: allEventsIndex,
          audioEvent: audioEventDetails, // This will be null for events without audio
      });
      });
  }
  console.debug(`DEBUG: [loadQueueFrom] Loaded ${audioState.audioQueue.length} events into queue.`);
  }