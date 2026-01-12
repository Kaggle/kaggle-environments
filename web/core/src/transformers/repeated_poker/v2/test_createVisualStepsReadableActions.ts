import { createVisualStepsFromRepeatedPokerReplay } from './repeatedPokerTransformerUtils.js';
import {
  PokerReplayInfoAgent,
  PokerReplayStepHistoryParsed,
  PokerReplayUniversalPokerJson,
} from './poker-replay-types.js';

const agents: PokerReplayInfoAgent[] = [
  { Name: 'Agent 0', ThumbnailUrl: '' },
  { Name: 'Agent 1', ThumbnailUrl: '' },
];

const createUniversalState = (
  overrides: Partial<PokerReplayUniversalPokerJson> = {}
): PokerReplayUniversalPokerJson => ({
  acpc_state: 'STATE:0::AsKs|QhJh\nSpent: [P0: 2  P1: 1  ]',
  best_five_card_hands: ['', ''],
  best_hand_rank_types: ['', ''],
  betting_history: '',
  blinds: [1, 2],
  board_cards: '',
  current_player: 1,
  odds: [0.5, 0.5],
  player_contributions: [2, 1],
  player_hands: ['AsKs', 'QhJh'],
  pot_size: 3,
  starting_stacks: [200, 200],
  ...overrides,
});

const beforeActionStep: PokerReplayStepHistoryParsed = {
  big_blind: 2,
  current_universal_poker_json: createUniversalState(),
  dealer: 1,
  hand_number: 1,
  hand_returns: [[0, 0]],
  max_num_hands: 1,
  prev_universal_poker_json: createUniversalState(),
  small_blind: 1,
  stacks: [200, 200],
  step: {
    action: {
      submission: 0,
      actionString: 'move=r5',
    },
    info: {},
    observation: {
      currentPlayer: 1,
      isTerminal: false,
      legalActionStrings: [],
      legalActions: [],
      observationString: '',
      playerId: 1,
      remainingOverageTime: 0,
      serializedGameAndState: '',
    },
    reward: null,
    status: 'NORMAL',
  },
  stepIndex: 0,
};

const afterActionOverrides: Partial<PokerReplayUniversalPokerJson> = {
  acpc_state: 'STATE:0:r5:AsKs|QhJh\nSpent: [P0: 2  P1: 5  ]',
  current_player: 0,
  player_contributions: [2, 5],
  pot_size: 7,
};

const afterActionStep: PokerReplayStepHistoryParsed = {
  big_blind: 2,
  current_universal_poker_json: createUniversalState(afterActionOverrides),
  dealer: 1,
  hand_number: 1,
  hand_returns: [[0, 0]],
  max_num_hands: 1,
  prev_universal_poker_json: createUniversalState(afterActionOverrides),
  small_blind: 1,
  stacks: [200, 200],
};

const visualSteps = createVisualStepsFromRepeatedPokerReplay([beforeActionStep, afterActionStep], agents);
const playerActionStep = visualSteps.find((step) => step.stepType === 'player-action');

if (!playerActionStep) {
  console.error('❌ player action step not generated');
}

if (playerActionStep) {
  const actingPlayerDisplay = playerActionStep.players[1].actionDisplayText;
  const nonActingPlayerDisplay = playerActionStep.players[0].actionDisplayText;

  const expectedActingPlayerDisplay = 'Raise 5';
  const expectedNonActingPlayerDisplay = '3 to call';

  if (
    actingPlayerDisplay === expectedActingPlayerDisplay &&
    nonActingPlayerDisplay === expectedNonActingPlayerDisplay
  ) {
    console.log('✅ player action step uses readable ACPC action');
  } else {
    console.error('❌ readable ACPC action mismatch');
    console.error(`   expected acting player: ${expectedActingPlayerDisplay}`);
    console.error(`   expected non-acting player: ${expectedNonActingPlayerDisplay}`);
    console.error(`   received acting player: ${actingPlayerDisplay}`);
    console.error(`   received non-acting player: ${nonActingPlayerDisplay}`);
  }
}
