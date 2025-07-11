import json
import random  # Added for random.choice
from enum import Enum
from os import path

from .game.actions import Action, EliminateProposalAction, VoteAction, HealAction, InspectAction, ChatAction, \
    NoOpAction, create_action
from .game.engine import Moderator, DetailedPhase
from .game.protocols import (
    DiscussionProtocol, VotingProtocol,
    RoundRobinDiscussion, SimultaneousMajority, ParallelDiscussion, SequentialVoting
)
from .game.records import AskDoctorSaveDataEntry, AskSeerRevealDataEntry, \
    AskWerewolfVotingDataEntry
from .game.roles import create_players_from_roles_and_ids
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
    player_id: str
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
    my_id = raw_obs['player_id']
    alive_players = raw_obs['alive_players']

    action = NoOpAction(actor_id=my_id, reasoning="There's nothing to be done.") # Default action

    if current_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS:
        if my_role == RoleConst.WEREWOLF:
            # Werewolves target other alive players. A smarter agent would parse history to find non-werewolves.
            # ActionType.NIGHT_KILL_VOTE
            history_entry = next((entry for entry in entries
                                  if isinstance(entry.data, AskWerewolfVotingDataEntry)), None)
            if history_entry:
                valid_targets = history_entry.data.valid_targets
                target_id = random.choice(valid_targets)
                action = EliminateProposalAction(actor_id=my_id, target_id=target_id, reasoning="I randomly chose one.")

        elif my_role == RoleConst.DOCTOR:
            # Doctors can save any alive player (including themselves)
            # ActionType.NIGHT_SAVE_TARGET
            history_entry = next((entry for entry in entries if isinstance(entry.data, AskDoctorSaveDataEntry)), None)

            if history_entry:
                valid_targets = history_entry.data.valid_candidates
                target_id = random.choice(valid_targets)
                action = HealAction(actor_id=my_id, target_id=target_id, reasoning="I randomly chose one to heal.")

        elif my_role == RoleConst.SEER:
            # Seers can inspect any alive player
            # ActionType.NIGHT_INSPECT_TARGET
            history_entry = next((entry for entry in entries if isinstance(entry.data, AskSeerRevealDataEntry)), None)

            if history_entry:
                valid_targets = history_entry.data.valid_candidates
                target_id = random.choice(valid_targets)
                action = InspectAction(actor_id=my_id, target_id=target_id, reasoning="I randomly chose one to inspect.")

    elif current_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT:
        # Only alive players can discuss
        if my_id in alive_players:
            action = ChatAction(
                actor_id=my_id,
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
        if my_id in alive_players:
            action = VoteAction(
                actor_id=my_id,
                target_id=random.choice(all_player_names),
                reasoning="I randomly chose one."
            )

    elif current_phase == DetailedPhase.GAME_OVER:
        action = {"action_type": ActionType.NO_OP.value}

    return action.serialize()


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


def _prepare_observation(sub_state, player_obj, game_state, detailed_phase):
    new_history_entries = player_obj.consume_messages()
    sub_state['observation']['new_history_entries'] = new_history_entries
    obs = WerewolfObservationModel(
        player_id=player_obj.id,
        role=player_obj.role.name,
        team=player_obj.role.team.value,
        is_alive=player_obj.alive,
        day=game_state.day_count,
        phase=detailed_phase.value,
        all_player_ids=game_state.all_player_ids,
        alive_players=[p.id for p in game_state.alive_players()],
        new_visible_announcements=[entry.description for entry in new_history_entries],
        new_visible_raw_data=[VisibleRawData.from_entry(entry) for entry in new_history_entries if entry.data],
        game_state_phase=game_state.phase.value
    )
    sub_state.observation["raw_observation"] = obs.model_dump()


def interpreter(state, env):
    """
    state: list of dictionaries, one for each agent.
           Each dict has: {observation, action, reward, status, info}
    env:   the kaggle_environments.Environment object itself.
    """

    # --- Initialize Moderator and GameState if it's the start of an episode ---
    if not hasattr(env, 'moderator') or env.done: # env.done is true after reset by Kaggle core
        num_players = len(state)

        roles_from_config = env.configuration.roles
        names_from_config = env.configuration.names

        # below checks for configuration consistency with agent count. If inconsistent, it will cause down stream subtle error.
        if len(roles_from_config) < num_players:
            raise ValueError(f"Configuration has {len(roles_from_config)} roles, but {num_players} agents are present.")
        if len(names_from_config) < num_players:
            raise ValueError(f"Configuration has {len(names_from_config)} names, but {num_players} agents are present.")

        players = create_players_from_roles_and_ids(role_strings=roles_from_config[:num_players], player_ids=names_from_config[:num_players])
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
    
    if is_game_done:
        # Determine winner based on game_state.history's GAME_END entry
        end_entry = game_state.get_history_by_type(entry_type=HistoryEntryType.GAME_END)[0]
        scores = end_entry.data.scores

        for i, player_id in enumerate(env.player_id_str_list):
            state[i].reward = scores[player_id]

    active_player_ids_after_advance = set(moderator.get_active_player_ids())

    for i in range(len(state)):
        player_id_str = env.player_ids_map[i]

        # skip if player not active and game is not done
        if player_id_str not in active_player_ids_after_advance and not is_game_done:
           state[i].status = 'INACTIVE'
           continue
        
        # set the status of active player to ACTIVE
        state[i].status = 'ACTIVE'
        player_obj = game_state.get_player_by_id(player_id_str)

        # Observation processing
        new_history_entries = player_obj.consume_messages()

        state[i]['observation']['new_history_entries'] = new_history_entries

        obs = WerewolfObservationModel(
            player_id=player_obj.id,
            role=player_obj.role.name,
            team=player_obj.role.team.value,
            is_alive=player_obj.alive,
            day=game_state.day_count,
            phase=moderator.detailed_phase.value,
            all_player_ids=game_state.all_player_ids,
            alive_players=[p.id for p in game_state.alive_players()],
            new_visible_announcements=[entry.description for entry in new_history_entries],
            new_visible_raw_data=[VisibleRawData.from_entry(entry) for entry in new_history_entries if entry.data],
            game_state_phase=game_state.phase.value
        )

        state[i].observation["raw_observation"] = obs.model_dump()

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
            game_end_entry = game_state.get_history_by_type(HistoryEntryType.GAME_END)[0]
            if game_end_entry and game_end_entry.data:
                current_info["winner_team"] = game_end_entry.data.winner_team
                current_info["win_reason"] = game_end_entry.data.reason
                current_info["scores"] = game_end_entry.data.scores
                current_info["survivors_until_last_round_and_role"] = game_end_entry.data.survivors_until_last_round_and_role
        state[i].info = current_info
    return state


def renderer(state, env):
    if not hasattr(env, 'moderator') or not hasattr(env, 'game_state'):
        return "Game not initialized by interpreter yet."

    game_state: GameState = env.game_state

    lines = []
    for entry in game_state.consume_messages():
        lines.append(entry.description)
    return "\n\n".join(lines)


def html_renderer():
    jspath = path.abspath(path.join(path.dirname(__file__), "werewolf.js"))
    with open(jspath, encoding="utf-8") as f:
        return f.read()


jsonpath = path.abspath(path.join(path.dirname(__file__), "werewolf.json"))
with open(jsonpath) as f:
    specification = json.load(f)