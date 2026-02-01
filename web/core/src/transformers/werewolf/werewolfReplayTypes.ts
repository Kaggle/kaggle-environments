import { BaseGamePlayer, ReplayData } from '../../types';

export interface WerewolfEvent {
  event_name: string;
  description?: string;
  originalDescription?: string;
  visible_in_ui?: boolean;
  kaggleStep?: number;
  dataType?: string;
  day?: number;
  phase?: string;
  created_at?: string;
  data?: {
    actor_id?: string;
    target_id?: string;
    reasoning?: string;
    message?: string;
    elected_player_id?: string;
    eliminated_player_id?: string;
    winner_team?: string;
    [key: string]: any;
  };
  [key: string]: any;
}

/**
 * Werewolf player extends BaseGamePlayer but uses string name as the identifier
 * (werewolf uses player_0, player_1, etc. as IDs instead of numeric IDs)
 */
export interface WerewolfPlayer extends Omit<BaseGamePlayer, 'id'> {
  id?: number; // Optional since werewolf uses string names
  role?: string;
  team?: string;
  isAlive?: boolean;
}

export interface WerewolfVisualizerData {
  allEvents: WerewolfEvent[];
  displayStepToAllEventsIndex: number[];
  allEventsIndexToDisplayStep: number[];
  eventToKaggleStep: number[];
  originalSteps: any[];
  /** Map of character name to agent config (with display_name for model name) */
  playerConfigMap: Record<string, { display_name?: string; thumbnail?: string; [key: string]: any }>;
}

/**
 * Werewolf step that conforms to BaseGameStep for use with ReasoningLogs.
 * Also includes the original step data and visualizerEvent for the 3D renderer.
 */
export interface WerewolfStep {
  step: number;
  players: WerewolfPlayer[];
  visualizerEvent?: WerewolfEvent;
  originalStepData?: any[];
}

export interface WerewolfProcessedReplay extends ReplayData<WerewolfStep[]> {
  visualizerData: WerewolfVisualizerData;
}
