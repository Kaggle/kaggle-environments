import json
import random  # Added for random.choice
from enum import Enum
from os import path
from typing import Dict, Optional, List, Any, Union, Tuple

from pydantic import BaseModel

from .game.actions import Action, VoteAction, HealAction, InspectAction, ChatAction, NoOpAction
# Game engine components for the new interpreter
from .game.engine import Moderator, DetailedPhase
from .game.protocols import (
    DiscussionProtocol, VotingProtocol,
    RoundRobinDiscussion, SimultaneousMajority, ParallelDiscussion, SequentialVoting
)
from .game.roles import Player, Werewolf, Villager, Seer, Doctor, RoleConst
from .game.states import GameState, HistoryEntryType


# my_kaggle_env.py
# Enums used by agents and action parser


class ActionType(str, Enum):
    NO_OP = "NO_OP"
    NIGHT_KILL_VOTE = "NIGHT_KILL_VOTE"
    NIGHT_SAVE_TARGET = "NIGHT_SAVE_TARGET"
    NIGHT_INSPECT_TARGET = "NIGHT_INSPECT_TARGET"
    DAY_DISCUSS = "DAY_DISCUSS"
    DAY_LYNCH_VOTE = "DAY_LYNCH_VOTE"


class WerewolfObservationModel(BaseModel):
    my_unique_name: str
    role: str
    team: str
    is_alive: bool
    day: int
    phase: str
    all_player_unique_names: str  # JSON string
    alive_players: List[bool]
    visible_history: Tuple[str, ...]
    action_prompt: str
    game_state_phase: str
    my_player_idx: int

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
    raw_aec_obs = obs.get('raw_aec_observation')

    # Default to NO_OP if observation is missing or agent cannot act
    if not raw_aec_obs:
        return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}

    current_phase = DetailedPhase(raw_aec_obs['phase'])
    my_role = RoleConst(raw_aec_obs['role'])

    all_player_names = json.loads(raw_aec_obs['all_player_unique_names'])
    my_unique_name = raw_aec_obs['my_unique_name']

    my_idx = all_player_names.index(my_unique_name)

    alive_player_indices = [i for i, status in enumerate(raw_aec_obs['alive_players']) if status == 1]

    action_to_take = {"action_type": ActionType.NO_OP.value} # Default action

    if current_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS:
        if my_role == RoleConst.WEREWOLF:
            # Werewolves target other alive players. A smarter agent would parse history to find non-werewolves.
            potential_targets = [idx for idx in alive_player_indices if idx != my_idx]
            if potential_targets:
                target_idx = random.choice(potential_targets)
                action_to_take = {"action_type": ActionType.NIGHT_KILL_VOTE.value, "target_idx": target_idx}

        elif my_role == RoleConst.DOCTOR:
            # Doctors can save any alive player (including themselves)
            if alive_player_indices:
                target_idx = random.choice(alive_player_indices)
                action_to_take = {"action_type": ActionType.NIGHT_SAVE_TARGET.value, "target_idx": target_idx}

        elif my_role == RoleConst.SEER:
            # Seers can inspect any alive player
            if alive_player_indices:
                target_idx = random.choice(alive_player_indices)
                action_to_take = {"action_type": ActionType.NIGHT_INSPECT_TARGET.value, "target_idx": target_idx}

    elif current_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT:
        if my_idx in alive_player_indices: # Only alive players can discuss
            messages = [
                "Hello everyone!", 
                "I have a strong feeling about someone.", 
                "Any information to share?", 
                "I am a simple Villager just trying to survive.", 
                "Let's think carefully before voting."
            ]
            
            if len(alive_player_indices) > 0:
                rand_player_for_msg_idx = random.choice(alive_player_indices)
                messages[1] = f"I think {all_player_names[rand_player_for_msg_idx]} is acting suspiciously."
                
                votable_for_message = [p_idx for p_idx in alive_player_indices if p_idx != rand_player_for_msg_idx]
                if votable_for_message:
                    rand_player_for_vote_msg_idx = random.choice(votable_for_message)
                    messages[4] = f"We should consider voting for {all_player_names[rand_player_for_vote_msg_idx]} today."
                elif len(alive_player_indices) == 1: 
                     messages[4] = "It seems I'm the only one left to talk to."

            action_to_take = {"action_type": ActionType.DAY_DISCUSS.value, "message": random.choice(messages)}

    elif current_phase == DetailedPhase.DAY_VOTING_AWAIT:
        if my_idx in alive_player_indices: # Only alive players can vote
            votable_targets = [p_idx for p_idx in alive_player_indices if p_idx != my_idx]
            if votable_targets:
                target_idx = random.choice(votable_targets)
                action_to_take = {"action_type": ActionType.DAY_LYNCH_VOTE.value, "target_idx": target_idx}
    
    elif current_phase == DetailedPhase.GAME_OVER:
        action_to_take = {"action_type": ActionType.NO_OP.value}
        
    if "target_idx" not in action_to_take:
        action_to_take["target_idx"] = None
    if "message" not in action_to_take:
        action_to_take["message"] = None
        
    return action_to_take

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
        raw_aec_obs = obs.get('raw_aec_observation')

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
        env.player_ids_map = {i: str(i) for i in range(num_players)} # map kaggle index to player ID string
        env.player_id_str_list = [str(i) for i in range(num_players)]

        # Simplified role assignment (can be made more complex like in WerewolfEnv.reset)
        # Ensure roles are from .game.roles
        roles_to_assign = []
        # Example: 1 WW, 1 Seer, 1 Doctor, rest Villagers
        if num_players >= 1: roles_to_assign.append(Werewolf())
        if num_players >= 2: roles_to_assign.append(Seer())
        if num_players >= 3: roles_to_assign.append(Doctor())
        while len(roles_to_assign) < num_players:
            roles_to_assign.append(Villager())
        
        if len(roles_to_assign) > num_players: # Should not happen with above logic
            roles_to_assign = roles_to_assign[:num_players]

        random.shuffle(roles_to_assign)

        players = [Player(id=env.player_id_str_list[i], role=roles_to_assign[i]) for i in range(num_players)]
        env.game_state = GameState(players=players, history={})

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
        # Night voting for werewolves often uses a simpler majority rule,
        # but can also be configured.
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
        # Moderator initializes its own _player_history_cursors

        # Initial advance to set up the first phase (e.g., NIGHT_START -> NIGHT_AWAIT_ACTIONS)
        env.moderator.advance({}) # Empty actions for the first transition

        for i in range(num_players):
            state[i].reward = 0
            state[i].info = {}
            # Initial observation and status will be set in the main update loop below

    moderator: Moderator = env.moderator
    game_state: GameState = env.game_state

    # 1. Collect and parse actions from Kaggle agents
    parsed_player_actions: Dict[str, Action] = {}
    active_player_ids_from_moderator = moderator.get_active_player_ids()

    for sub_state, player in zip(state, game_state.players):
        player_id_str = player.id
        if player_id_str in active_player_ids_from_moderator and sub_state.status == "ACTIVE":
            raw_action_from_agent = sub_state.action
            actor_role = game_state.get_player_by_id(player_id_str).role.name
            parsed_action = _parse_agent_action_to_engine_action(
                player_id_str, raw_action_from_agent, env.player_id_str_list,
                moderator.detailed_phase, actor_role
            )
            if parsed_action:
                parsed_player_actions[player_id_str] = parsed_action

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
        player_obj = game_state.get_player_by_id(player_id_str)

        # Observation processing
        new_history_entries, new_cursor_pos = moderator.get_observation(player_id_str)
        env.player_full_visible_history_cache.setdefault(player_id_str, []).extend(new_history_entries)
        moderator.update_player_cursor(player_id_str, new_cursor_pos)

        current_player_full_log = env.player_full_visible_history_cache[player_id_str]
        visible_history_descs = [entry.description for entry in current_player_full_log[-MAX_VISIBLE_HISTORY_ITEMS:]]
        visible_history_descs.extend([""] * (MAX_VISIBLE_HISTORY_ITEMS - len(visible_history_descs)))

        latest_prompt = "No specific prompt. It's your turn to act if active."
        for entry in reversed(current_player_full_log): # Search in player's own cached history
            if entry.entry_type == HistoryEntryType.MODERATOR_ANNOUNCEMENT and player_id_str in entry.visible_to:
                latest_prompt = entry.description
                break

        obs_data = {
            "my_unique_name": player_id_str,
            "role": player_obj.role.name, # RoleConst enum value (string)
            "team": player_obj.role.team.value, # Team enum value (string)
            "is_alive": player_obj.alive,
            "day": game_state.day_count,
            "phase": moderator.detailed_phase.value, # DetailedPhase enum value (string)
            "all_player_unique_names": json.dumps(env.player_id_str_list), # JSON string list of player IDs
            "alive_players": [p.alive for p in game_state.players], # List of booleans
            "visible_history": tuple(visible_history_descs),
            "action_prompt": latest_prompt,
            "game_state_phase": game_state.phase.value, # Overall Day/Night from GameState
            "my_player_idx": i,
        }
        state[i].observation = {"raw_aec_observation": obs_data} # Nest to match agent access pattern

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