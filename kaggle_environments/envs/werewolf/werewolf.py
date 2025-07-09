import json
import random  # Added for random.choice
from enum import Enum
from os import path
from typing import Dict, Optional, List, Any, Union, Tuple

from pydantic import BaseModel
from .game.actions import Action, EliminateProposalAction, VoteAction, HealAction, InspectAction, ChatAction, \
    NoOpAction, create_action
from .game.engine import Moderator, DetailedPhase
from .game.protocols import (
    DiscussionProtocol, VotingProtocol,
    RoundRobinDiscussion, SimultaneousMajority, ParallelDiscussion, SequentialVoting
)
from .game.roles import RoleConst, create_players_from_roles_and_ids
from .game.states import *


# my_kaggle_env.py
# Enums used by agents and action parser


class ActionType(str, Enum):
    NO_OP = "NO_OP"
    NIGHT_KILL_VOTE = "NIGHT_KILL_VOTE"
    NIGHT_SAVE_TARGET = "NIGHT_SAVE_TARGET"
    NIGHT_INSPECT_TARGET = "NIGHT_INSPECT_TARGET"
    DAY_DISCUSS = "DAY_DISCUSS"
    DAY_LYNCH_VOTE = "DAY_LYNCH_VOTE"


class VisibleRawData(BaseModel):
    data_type: str
    json_str: str
    """json dump"""

    @classmethod
    def from_entry(cls, entry: DataEntry):
        return cls(data_type=entry.data.__class__.__name__, json_str=entry.data.model_dump_json())


class WerewolfObservationModel(BaseModel):
    my_unique_name: str
    role: str
    team: str
    is_alive: bool
    day: int
    phase: str
    all_player_ids: List[str]
    alive_players: List[str]
    new_visible_announcements: List[str]
    new_visible_raw_data: List[VisibleRawData]
    game_state_phase: str

    def get_human_readable(self) -> str:
        # This is a placeholder implementation. A real implementation would format this nicely.
        return json.dumps(self.model_dump(), indent=2)


MAX_VISIBLE_HISTORY_ITEMS = 20 # Max number of history items in agent observation


# --- Protocol Factory ---
PROTOCOL_REGISTRY = {
    "discussion": {
        "RoundRobinDiscussion": {"class": RoundRobinDiscussion, "default_params": {"max_rounds": 1}},
        "ParallelDiscussion": {"class": ParallelDiscussion, "default_params": {"ticks": 3}},
        # Add other discussion protocols here if needed
    },
    "voting": {
        "SimultaneousMajority": {"class": SimultaneousMajority, "default_params": {}},
        "SequentialVoting": {"class": SequentialVoting, "default_params": {}},
        # Add other voting protocols here if needed
    }
}

DEFAULT_DISCUSSION_PROTOCOL_NAME = "RoundRobinDiscussion"
DEFAULT_VOTING_PROTOCOL_NAME = "SimultaneousMajority"


def create_protocol_from_config(
    config: Any, # env.configuration
    protocol_config_key: str, # e.g., "discussion_protocol"
    protocol_type: str, # "discussion" or "voting"
    default_protocol_name: str
) -> Union[DiscussionProtocol, VotingProtocol]:
    protocol_config = getattr(config, protocol_config_key, {})
    protocol_name = protocol_config.get("name", default_protocol_name)
    user_params = protocol_config.get("params", {})

    registry_for_type = PROTOCOL_REGISTRY[protocol_type]
    protocol_info = registry_for_type.get(protocol_name)
    if not protocol_info:
        print(f"Warning: Protocol '{protocol_name}' not found in {protocol_type} registry. Using default '{default_protocol_name}'.")
        protocol_info = registry_for_type[default_protocol_name]
    protocol_class, default_params_dict = protocol_info["class"], protocol_info["default_params"]
    final_params = {**default_params_dict, **user_params}
    return protocol_class(**final_params)


def random_agent(obs):
    raw_obs = obs.get('raw_observation')
    entries = obs.get('new_history_entries')

    # Default to NO_OP if observation is missing or agent cannot act
    if not raw_obs or not entries:
        return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}

    current_phase = DetailedPhase(raw_obs['phase'])
    my_role = RoleConst(raw_obs['role'])

    all_player_names = raw_obs['all_player_ids']
    my_unique_name = raw_obs['my_unique_name']

    my_idx = all_player_names.index(my_unique_name)

    alive_player_indices = [i for i, status in enumerate(raw_obs['alive_players']) if status == 1]

    action = NoOpAction(reasoning="There's nothing to be done.") # Default action

    if current_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS:
        if my_role == RoleConst.WEREWOLF:
            # Werewolves target other alive players. A smarter agent would parse history to find non-werewolves.
            # ActionType.NIGHT_KILL_VOTE
            history_entry = next((entry for entry in entries
                                  if isinstance(entry.data, AskWerewolfVotingDataEntry)), None)
            if history_entry:
                valid_targets = history_entry.data.valid_targets
                target_id = random.choice(valid_targets)
                action = EliminateProposalAction(target_id=target_id, reasoning="I randomly chose one.")

        elif my_role == RoleConst.DOCTOR:
            # Doctors can save any alive player (including themselves)
            # ActionType.NIGHT_SAVE_TARGET
            history_entry = next((entry for entry in entries if isinstance(entry.data, AskDoctorSaveDataEntry)), None)

            if history_entry:
                valid_targets = history_entry.data.valid_candidates
                target_id = random.choice(valid_targets)
                action = HealAction(target_id=target_id, reasoning="I randomly chose one to heal.")

        elif my_role == RoleConst.SEER:
            # Seers can inspect any alive player
            # ActionType.NIGHT_INSPECT_TARGET
            history_entry = next((entry for entry in entries if isinstance(entry.data, AskSeerRevealDataEntry)), None)

            if history_entry:
                valid_targets = history_entry.data.valid_candidates
                target_id = random.choice(valid_targets)
                action = InspectAction(target_id=target_id, reasoning="I randomly chose one to inspect.")

    elif current_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT:
        # Only alive players can discuss
        if my_idx in alive_player_indices:
            action = ChatAction(
                message=random.choice([
                    "Hello everyone!",
                    "I have a strong feeling about someone.",
                    "Any information to share?",
                    "I am a simple Villager just trying to survive.",
                    "Let's think carefully before voting."
                ]),
                reasoning="I randomly chose one message."
            )

    elif current_phase == DetailedPhase.DAY_VOTING_AWAIT:
        # Only alive players can vote
        if my_idx in alive_player_indices:
            # ActionType.DAY_LYNCH_VOTE
            valid_targets_idx = [idx for idx in alive_player_indices if idx != my_idx]
            if valid_targets_idx:
                action = VoteAction(
                    target_id=random.choice(all_player_names),
                    reasoning="I randomly chose one."
                )

    elif current_phase == DetailedPhase.GAME_OVER:
        action = {"action_type": ActionType.NO_OP.value}

    return action.serialize()

# Helper function to parse agent's dict action to engine.Action
def _parse_agent_action_to_engine_action(
    actor_id_str: str,
    raw_agent_action: dict,
    all_player_id_strs: list[str],
    current_detailed_phase: DetailedPhase,
    actor_role_name: RoleConst
) -> Optional[Action]:

    if not raw_agent_action or not isinstance(raw_agent_action, dict):
        return NoOpAction(actor_id=actor_id_str, reasoning="No action provided by agent")

    action_type_val = raw_agent_action.get("action_type") # This is ActionType.value (a string)
    target_idx = raw_agent_action.get("target_idx")
    message = raw_agent_action.get("message")

    target_id_str = None
    if target_idx is not None:
        if target_idx == -1: # Abstain or no target convention
            target_id_str = "-1"
        elif 0 <= target_idx < len(all_player_id_strs):
            target_id_str = all_player_id_strs[target_idx]
        else: # Invalid index
            return NoOpAction(actor_id=actor_id_str, reason=f"Invalid target_idx: {target_idx}")

    # Phase-aware and Role-aware parsing
    if current_detailed_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS:
        if actor_role_name == RoleConst.DOCTOR:
            if action_type_val == ActionType.NIGHT_SAVE_TARGET.value and target_id_str and target_id_str != "-1":
                return HealAction(actor_id=actor_id_str, target_id=target_id_str)
        elif actor_role_name == RoleConst.SEER:
            if action_type_val == ActionType.NIGHT_INSPECT_TARGET.value and target_id_str and target_id_str != "-1":
                return InspectAction(actor_id=actor_id_str, target_id=target_id_str)
        elif actor_role_name == RoleConst.WEREWOLF:
            # Werewolf night vote action is VoteAction
            if action_type_val == ActionType.NIGHT_KILL_VOTE.value and target_id_str:
                return VoteAction(actor_id=actor_id_str, target_id=target_id_str)

    elif current_detailed_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT:
        if action_type_val == ActionType.DAY_DISCUSS.value and message:
            return ChatAction(actor_id=actor_id_str, message=str(message))

    elif current_detailed_phase == DetailedPhase.DAY_VOTING_AWAIT:
        if action_type_val == ActionType.DAY_LYNCH_VOTE.value and target_id_str:
            return VoteAction(actor_id=actor_id_str, target_id=target_id_str)

    if action_type_val == ActionType.NO_OP.value:
        return NoOpAction(actor_id=actor_id_str)

    # Fallback if action_type_val didn't match any valid action for the current phase/role
    return NoOpAction(actor_id=actor_id_str, reason=f"Action '{action_type_val}' not applicable or invalid for phase '{current_detailed_phase.value}' and role '{actor_role_name.value}'")


# This function is part of the skeleton and retained as a placeholder.
def dummy_inference_endpoint(prompt):
    # In a real scenario, this would query an LLM.
    # For testing, we can make it return a valid JSON action string.
    # Example: return '{"action_type": "NO_OP"}'
    return '{"action_type": "NO_OP", "message": "dummy action"}'


endpoints = {'dummy_llm': dummy_inference_endpoint}


class LLMAgent:
    def __init__(self, model_name="dummy_llm", system_prompt="You are a helpful assistant playing Werewolf."):
        """
        Initializes the LLMAgent.
        Args:
            model_name (str): Identifier for the LLM model (currently conceptual).
            system_prompt (str): A system prompt to guide the LLM's behavior.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.memory = [] # Stores a history of observations or processed information
        self.inferencer = endpoints[model_name]
    
    def parse_llm_response_to_action(self, llm_response_str: str) -> dict:
        """
        Parses a JSON string from an LLM into a valid game action dictionary.

        Args:
            llm_response_str: The JSON string response from the LLM.
                            Expected format: {"action_type": "ACTION_NAME_STR", "target_idx": int_or_null, "message": "str_or_null"}

        Returns:
            A dictionary representing the game action, or a NO_OP action if parsing fails.
        """
        try:
            action_data = json.loads(llm_response_str)
            if not isinstance(action_data, dict):
                raise ValueError("LLM response is not a JSON object.")

            action_type_str = action_data.get("action_type")
            if not action_type_str or not hasattr(ActionType, action_type_str):
                raise ValueError(f"Invalid or missing 'action_type': {action_type_str}")

            action_type_enum_val = ActionType[action_type_str].value
            target_idx = action_data.get("target_idx") # Can be None
            message = action_data.get("message")     # Can be None

            # Basic type check for target_idx if present
            if target_idx is not None and not isinstance(target_idx, int):
                target_idx = None # Or raise error, but defaulting to None is safer for NO_OP fallback

            return {"action_type": action_type_enum_val, "target_idx": target_idx, "message": message}

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Error parsing LLM response '{llm_response_str}': {e}. Defaulting to NO_OP.")
            return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}
    
    def __call__(self, obs):
        """
        Processes an observation, updates memory, and decides on an action.
        Currently, it only stores the observation and returns a NO_OP action.
        """
        raw_aec_obs = obs.get('raw_observation')

        if not raw_aec_obs:
            # Default action if no observation is available
            return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}

        # Convert raw observation to a more readable format.
        # If WerewolfObservationModel instantiation or get_human_readable fails,
        # the error will propagate as per the "no try-except" constraint.
        pydantic_obs = WerewolfObservationModel(**raw_aec_obs)
        human_readable_obs = pydantic_obs.get_human_readable()
        
        # Update memory
        self.memory.append(human_readable_obs)

        # --- Placeholder for actual LLM interaction (conceptual) ---
        current_prompt = f"{self.system_prompt}\n\nObservation History:\n{json.dumps(self.memory, indent=2)}\n\nWhat is your action?"
        llm_response_action_str = self.inferencer(current_prompt)
        action_to_take = self.parse_llm_response_to_action(llm_response_action_str)
        
        return action_to_take
    

agents = {"random": random_agent, "dummy_llm": LLMAgent('dummy_llm')}


def interpreter(state, env):
    """
    state: list of dictionaries, one for each agent.
           Each dict has: {observation, action, reward, status, info}
    env:   the kaggle_environments.Environment object itself.
    """

    # --- Initialize Moderator and GameState if it's the start of an episode ---
    if not hasattr(env, 'moderator') or env.done: # env.done is true after reset by Kaggle core
        num_players = len(state)

        players = create_players_from_roles_and_ids(role_strings=env.configuration.roles, player_ids=env.configuration.names)
        env.game_state = GameState(players=players, history={})

        env.player_ids_map = {i: p.id for i, p in enumerate(players)}
        env.player_id_str_list = [p.id for p in players]

        # Initialize protocols from configuration or defaults
        discussion_protocol = create_protocol_from_config(
            env.configuration,
            "discussion_protocol",
            "discussion", DEFAULT_DISCUSSION_PROTOCOL_NAME
        )
        day_voting_protocol = create_protocol_from_config(
            env.configuration,
            "day_voting_protocol",
            "voting", DEFAULT_VOTING_PROTOCOL_NAME
        )
        # Night voting can be configured.
        night_voting_protocol = create_protocol_from_config(
            env.configuration,
            "werewolf_night_vote_protocol",
            "voting", DEFAULT_VOTING_PROTOCOL_NAME # Default to same as day voting if not specified
        )

        print(f"Interpreter: Using Discussion: {type(discussion_protocol).__name__}, Day Voting: {type(day_voting_protocol).__name__}, Night WW Voting: {type(night_voting_protocol).__name__}")

        env.moderator = Moderator(
            state=env.game_state,
            discussion=discussion_protocol,
            day_voting=day_voting_protocol,
            night_voting=night_voting_protocol
        )

        env.player_full_visible_history_cache = {p_id: [] for p_id in env.player_id_str_list}

    moderator: Moderator = env.moderator
    game_state: GameState = env.game_state

    # 1. Collect and parse actions from Kaggle agents
    parsed_player_actions: Dict[str, Action] = {}
    active_player_ids_from_moderator = moderator.get_active_player_ids()

    for sub_state, player in zip(state, game_state.players):
        player_id_str = player.id
        if player_id_str in active_player_ids_from_moderator and sub_state.status == "ACTIVE":
            serialized_action = sub_state.action
            if serialized_action:
                parsed_player_actions[player_id_str] = create_action(serialized_action)

    # 2. Advance the Moderator
    moderator.advance(parsed_player_actions)

    # 3. Update Kaggle state (observations, rewards, statuses)
    is_game_done = moderator.is_game_over()
    
    agent_rewards_this_step: Dict[str, float] = {}
    if is_game_done:
        # Determine winner based on game_state.history's GAME_END entry
        # The moderator._determine_and_log_winner() handles logging this.
        game_end_entry = next((e for day_hist in game_state.history.values() for e in day_hist if e.entry_type == HistoryEntryType.GAME_END), None)
        winning_team_str = None
        if game_end_entry and game_end_entry.data:
            winning_team_str = game_end_entry.data.get("winner_team")

        for p in game_state.players:
            if winning_team_str == "Draw":
                 agent_rewards_this_step[p.id] = 0.0
            elif p.role.team.value == winning_team_str:
                 agent_rewards_this_step[p.id] = 1.0
            else:
                 agent_rewards_this_step[p.id] = -1.0
    
    active_player_ids_after_advance = moderator.get_active_player_ids()

    for i in range(len(state)):
        player_id_str = env.player_ids_map[i]

        # skip if player not active
        #if player_id_str not in active_player_ids_after_advance:
        #    continue
        
        # set the status of active player to ACTIVE
        #state[i]['status'] = 'ACTIVE'
        print(f"Player {player_id_str} status after advance: {state[i]['status']}")


        player_obj = game_state.get_player_by_id(player_id_str)

        # Observation processing
        new_history_entries, new_cursor_pos = moderator.get_observation(player_id_str)
        env.player_full_visible_history_cache.setdefault(player_id_str, []).extend(new_history_entries)
        moderator.update_player_cursor(player_id_str, new_cursor_pos)

        state[i]['observation']['new_history_entries'] = new_history_entries

        obs = WerewolfObservationModel(
            my_unique_name=player_id_str,
            role=player_obj.role.name,
            team=player_obj.role.team.value,
            is_alive=player_obj.alive,
            day=game_state.day_count,
            phase=moderator.detailed_phase.value,
            all_player_ids=env.player_id_str_list,
            alive_players=[p.id for p in game_state.alive_players()],
            new_visible_announcements=[entry.description for entry in new_history_entries],
            new_visible_raw_data=[VisibleRawData.from_entry(entry) for entry in new_history_entries if entry.data],
            game_state_phase=game_state.phase.value
        )

        state[i].observation["raw_observation"] = obs.model_dump()

        # Reward
        state[i].reward = agent_rewards_this_step.get(player_id_str, 0.0)

        # Status
        if is_game_done:
            state[i].status = "DONE"
        elif player_id_str in active_player_ids_after_advance:
            state[i].status = "ACTIVE"
        else:
            state[i].status = "INACTIVE"
        
        # Info
        current_info = {}
        if is_game_done:
            game_end_entry = next((e for day_hist in game_state.history.values() for e in day_hist if e.entry_type == HistoryEntryType.GAME_END), None)
            if game_end_entry and game_end_entry.data:
                current_info["winner_team"] = game_end_entry.data.get("winner_team")
                current_info["win_reason"] = game_end_entry.data.get("reason")
        state[i].info = current_info

    return state


def renderer(state, env):
    if not hasattr(env, 'moderator') or not hasattr(env, 'game_state'):
        return "Game not initialized by interpreter yet."

    moderator: Moderator = env.moderator
    game_state: GameState = env.game_state
    
    lines = []
    lines.append(f"--- Werewolf Game ---")
    lines.append(f"Day: {game_state.day_count}, Game Phase: {game_state.phase.value} (Detailed: {moderator.detailed_phase.value})")
    lines.append("Players:")
    for p_obj in game_state.players:
        status = "Alive" if p_obj.alive else "Dead"
        # For debugging, show roles. In a real game, this might be hidden or player-specific.
        lines.append(f"  Player {p_obj.id} ({p_obj.role.name}, Team: {p_obj.role.team.value}): {status}")

    lines.append("\nRecent Public History (last 10):")
    public_history_entries = []
    for day_num in sorted(game_state.history.keys()):
        for entry in game_state.history[day_num]:
            if entry.public:
                public_history_entries.append(
                    f"  Day {entry.day} [{entry.phase.value}] ({entry.entry_type.value}): {entry.description}"
                )
    
    for entry_line in public_history_entries[-10:]: # Display last 10 public entries
        lines.append(entry_line)

    if not public_history_entries:
        lines.append("  No public history events yet.")
    
    if moderator.is_game_over():
        lines.append("\n--- GAME OVER ---")
        game_end_entry = next((e for day_hist in game_state.history.values() for e in day_hist if e.entry_type == HistoryEntryType.GAME_END), None)
        if game_end_entry:
            lines.append(f"  {game_end_entry.description}")
            if game_end_entry.data:
                lines.append(f"  Reason: {game_end_entry.data.get('reason')}")
        else:
            lines.append("  Winner determination pending or not logged in history.")

    return "\n".join(lines)


def html_renderer():
    jspath = path.abspath(path.join(path.dirname(__file__), "werewolf.js"))
    with open(jspath, encoding="utf-8") as f:
        return f.read()


jsonpath = path.abspath(path.join(path.dirname(__file__), "werewolf.json"))
with open(jsonpath) as f:
    specification = json.load(f)