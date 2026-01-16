import json
import logging
import random
from os import path
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from kaggle_environments.envs.werewolf.game.consts import DetailedPhase, EnvInfoKeys, PerceivedThreatLevel

from .game.actions import (
    Action,
    BidAction,
    ChatAction,
    HealAction,
    InspectAction,
    NoOpAction,
    VoteAction,
    create_action,
)
from .game.base import PlayerID
from .game.consts import RoleConst
from .game.engine import Moderator
from .game.protocols.factory import create_protocol
from .game.records import WerewolfObservationModel, get_raw_observation, set_raw_observation
from .game.roles import create_players_from_agents_config
from .game.states import EventName, GameState, get_last_action_request
from .harness.base import LLMCostTracker, LLMWerewolfAgent

logger = logging.getLogger(__name__)

# --- Protocol Factory ---
DEFAULT_DISCUSSION_PROTOCOL_NAME = "RoundRobinDiscussion"
DEFAULT_VOTING_PROTOCOL_NAME = "SimultaneousMajority"
DEFAULT_BIDDING_PROTOCOL_NAME = "UrgencyBiddingProtocol"


class AgentCost(BaseModel):
    total_cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AgentCostSummary(BaseModel):
    agent_config: Dict
    costs: AgentCost = Field(default_factory=AgentCost)
    data: Optional[LLMCostTracker] = None


class CostSummary(BaseModel):
    cost_per_agent: List[AgentCostSummary] = Field(default_factory=list)
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0


_PERCEIVED_THREAT_LEVELS = [item.value for item in PerceivedThreatLevel]


def random_agent(obs):
    raw_obs = get_raw_observation(obs)

    entries = raw_obs.new_player_event_views
    current_phase = DetailedPhase(raw_obs.detailed_phase)
    my_role = raw_obs.role
    all_player_names = raw_obs.all_player_ids
    my_id = raw_obs.player_id
    alive_players = raw_obs.alive_players
    day = raw_obs.day
    phase = raw_obs.game_state_phase
    common_args = {"day": day, "phase": phase, "actor_id": my_id}

    action = NoOpAction(**common_args, reasoning="There's nothing to be done.")  # Default action
    threat_level = random.choice(_PERCEIVED_THREAT_LEVELS)

    if current_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS:
        if my_role == RoleConst.WEREWOLF:
            history_entry = get_last_action_request(entries, EventName.VOTE_REQUEST)
            if history_entry:
                valid_targets = history_entry.data.get("valid_targets")
                if valid_targets:
                    target_id = random.choice(valid_targets)
                    action = VoteAction(
                        **common_args,
                        target_id=target_id,
                        reasoning="I randomly chose one.",
                        perceived_threat_level=threat_level,
                    )

        elif my_role == RoleConst.DOCTOR:
            history_entry = get_last_action_request(entries, EventName.HEAL_REQUEST)
            if history_entry:
                valid_targets = history_entry.data["valid_candidates"]
                if valid_targets:
                    target_id = random.choice(valid_targets)
                    action = HealAction(
                        **common_args,
                        target_id=target_id,
                        reasoning="I randomly chose one to heal.",
                        perceived_threat_level=threat_level,
                    )

        elif my_role == RoleConst.SEER:
            history_entry = get_last_action_request(entries, EventName.INSPECT_REQUEST)
            if history_entry:
                valid_targets = history_entry.data["valid_candidates"]
                if valid_targets:
                    target_id = random.choice(valid_targets)
                    action = InspectAction(
                        **common_args,
                        target_id=target_id,
                        reasoning="I randomly chose one to inspect.",
                        perceived_threat_level=threat_level,
                    )

    elif current_phase in [DetailedPhase.DAY_BIDDING_AWAIT, DetailedPhase.DAY_CHAT_AWAIT]:
        if current_phase == DetailedPhase.DAY_BIDDING_AWAIT:
            if my_id in alive_players:
                action = BidAction(
                    **common_args,
                    amount=random.randint(1, 4),
                    reasoning="I am bidding randomly.",
                    perceived_threat_level=threat_level,
                )
        else:  # It's a chat turn (DAY_CHAT_AWAIT)
            if my_id in alive_players:
                action = ChatAction(
                    **common_args,
                    message=random.choice(
                        [
                            "Hello everyone!",
                            f"I suspect {random.choice(all_player_names)}.",
                            "Any information to share?",
                            "I am a simple Villager just trying to survive.",
                            "Let's think carefully before voting.",
                        ]
                    ),
                    reasoning="I randomly chose one message.",
                    perceived_threat_level=threat_level,
                )

    elif current_phase == DetailedPhase.DAY_VOTING_AWAIT:
        if my_id in alive_players:
            # A real agent would parse the prompt for valid targets
            valid_targets = [p_id for p_id in alive_players if p_id != my_id]
            if valid_targets:
                action = VoteAction(
                    **common_args,
                    target_id=random.choice(valid_targets),
                    reasoning="I randomly chose one.",
                    perceived_threat_level=threat_level,
                )

    return action.serialize()


FIXED_MESSAGE = "I am a simple villager."
FIXED_REASONING = "I am going to do one fixed thing."


def deterministic_agent(obs):
    raw_obs = get_raw_observation(obs)

    entries = raw_obs.new_player_event_views
    current_phase = DetailedPhase(raw_obs.detailed_phase)
    my_role = raw_obs.role
    my_id = raw_obs.player_id
    alive_players = raw_obs.alive_players
    day = raw_obs.day
    phase = raw_obs.game_state_phase
    common_args = {"day": day, "phase": phase, "actor_id": my_id}

    action = NoOpAction(**common_args, reasoning="There's nothing to be done.")  # Default action
    threat_level = random.choice(_PERCEIVED_THREAT_LEVELS)

    if current_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS:
        if my_role == RoleConst.WEREWOLF:
            history_entry = get_last_action_request(entries, EventName.VOTE_REQUEST)
            if history_entry:
                valid_targets = history_entry.data.get("valid_targets")
                if valid_targets:
                    # always select first valid
                    target_id = valid_targets[0]
                    action = VoteAction(
                        **common_args,
                        target_id=target_id,
                        reasoning=FIXED_REASONING,
                        perceived_threat_level=threat_level,
                    )

        elif my_role == RoleConst.DOCTOR:
            history_entry = get_last_action_request(entries, EventName.HEAL_REQUEST)
            if history_entry:
                valid_targets = history_entry.data["valid_candidates"]
                if valid_targets:
                    target_id = valid_targets[0]
                    action = HealAction(
                        **common_args,
                        target_id=target_id,
                        reasoning=FIXED_REASONING,
                        perceived_threat_level=threat_level,
                    )

        elif my_role == RoleConst.SEER:
            history_entry = get_last_action_request(entries, EventName.INSPECT_REQUEST)
            if history_entry:
                valid_targets = history_entry.data["valid_candidates"]
                if valid_targets:
                    target_id = valid_targets[0]
                    action = InspectAction(
                        **common_args,
                        target_id=target_id,
                        reasoning=FIXED_REASONING,
                        perceived_threat_level=threat_level,
                    )

    elif current_phase in [DetailedPhase.DAY_BIDDING_AWAIT, DetailedPhase.DAY_CHAT_AWAIT]:
        if current_phase == DetailedPhase.DAY_BIDDING_AWAIT:
            if my_id in alive_players:
                action = BidAction(
                    **common_args,
                    amount=4,
                    reasoning=FIXED_REASONING,
                    perceived_threat_level=threat_level,
                )
        else:  # It's a chat turn (DAY_CHAT_AWAIT)
            if my_id in alive_players:
                action = ChatAction(
                    **common_args,
                    message=FIXED_MESSAGE,
                    reasoning=FIXED_REASONING,
                    perceived_threat_level=threat_level,
                )

    elif current_phase == DetailedPhase.DAY_VOTING_AWAIT:
        if my_id in alive_players:
            # A real agent would parse the prompt for valid targets
            valid_targets = [p_id for p_id in alive_players if p_id != my_id]
            if valid_targets:
                action = VoteAction(
                    **common_args,
                    target_id=valid_targets[0],
                    reasoning=FIXED_REASONING,
                    perceived_threat_level=threat_level,
                )

    return action.serialize()


class AgentFactoryWrapper:
    """
    A wrapper that creates and manages separate agent instances for each player.
    This is necessary for stateful agents to be used in the agent registry,
    preventing them from sharing state (like memory or history) across different players.
    """

    def __init__(self, agent_class, **kwargs):
        self._agent_class = agent_class
        self._shared_kwargs = kwargs
        self._kwargs = {}  # store configs of individual agents
        self._instances = {}
        self._agent_configs = None

    @property
    def agent_class(self):
        return self._agent_class

    def get_instance(self, player_id: PlayerID):
        return self._instances.get(player_id)

    def __call__(self, obs, config):
        """
        The main callable method for the agent. It routes the call to the correct
        player-specific agent instance.
        """
        raw_obs = get_raw_observation(obs)
        player_id = raw_obs.player_id  # get the current active player id

        if not player_id:
            # This could happen on initial steps or for an inactive agent.
            # Returning a NO_OP action is a safe fallback.
            return NoOpAction(
                day=raw_obs.day,
                phase=raw_obs.game_state_phase,
                actor_id="unknown_fallback",
                reasoning="AgentFactoryWrapper: No player_id found in observation.",
            ).serialize()

        if not self._agent_configs:
            self._agent_configs = {agent_config.id: agent_config for agent_config in config.agents}

        if player_id not in self._instances:
            # Create a new agent instance for this player
            self._kwargs[player_id] = {"agent_config": self._agent_configs.get(player_id)}
            self._instances[player_id] = self._agent_class(**self._shared_kwargs, **self._kwargs[player_id])
        return self._instances[player_id](obs)

    def reset(self):
        self._instances.clear()


# --- Agent Registry ---
LLM_SYSTEM_PROMPT = "You are a master strategist playing the game of Werewolf. Your goal is to win. You win as a team and not as individuals."


# *Package variable required by Kaggle Environments framework*
# These are base agents that the calling framework can choose from
# Provides a random_agent for testing and a convenient default 'llm' agent.

agents = {
    "random": random_agent,
    "deterministic": deterministic_agent,
    # "llm": AgentFactoryWrapper(
    #     LLMWerewolfAgent,
    #     model_name=getenv("WEREWOLF_LLM_MODEL", "gemini/gemini-2.5-pro"),
    #     system_prompt=LLM_SYSTEM_PROMPT,
    # ),
}


def register_agents(agent_dict: Dict[str, Callable]):
    agents.update(agent_dict)


def log_error(status_code, state, env):
    invalid_action = any(player_state["status"] == status_code for player_state in state)
    if invalid_action:
        logger.error(f"{status_code} DETECTED")
        for i, player_state in enumerate(state):
            if player_state["status"] == status_code:
                player = env.game_state.players[i]
                logger.error(
                    f"player.id={player.id}, player.agent.agent_id={player.agent.agent_id} "
                    f"returns action with status code {status_code}."
                )
    return invalid_action


def interpreter(state, env):
    """
    * Required interface function for kaggle environments package *

    This is the primary interface for the kaggle environment (kEnv) to step game forward.
    Briefly flow of logic is:
    Initialization - kEnv creates werewolf object and chooses players. Schema definition for
    this is in werewolf.json
    1) kEnv calls interpreter() with current game state recorded in env.game_state
    2) interpreter() reads game state and any new player actions and updates
       the games state based on those actions and flow of the game to env.game_state.
    3) interpreter() writes events to history data and also writes events about
       state change in the game to env.game_state and returns back to kEnv
    4) kEnv parses out the relevant game events via agent logic in harness/base.py,
       constructs final prompt, and performs external API calls for models and records back
       to env.game_state
    Go back to 1 and continue

    For example - consider discussion and voting by villagers. werewolf.interpreter()
    updates phase and writes history entry that solicits players for discussion.
    kEnv calls agents to get their discussion and writes them to the history/game state.
    kEnv then calls interpreter() that then updates game phase and writes history entry soliciting
    votes for exile. kEnv then calls agents and associated models to get their votes and writes
    responses to game state. env then calls interpreter() and moderator collects votes, determine
    who was exiled, performs that action and advances game phase and game state.
    And so on...

    Note - The UI is also updated after each call to interpreter() as that is the tick unit
    for the game.

    Note - env framework assumes that there is an action to be done by player, but
    for werewolf there are places where moderator is the one taking the action (e.g.
    counting votes and performing exile) so some game 'ticks' are larger than others.

    state: list of dictionaries, one for each agent.
           Each dict has: {observation, action, reward, status, info}
    env:   the kaggle_environments.Environment object itself including the env.game_state
    """
    agent_error = False
    for status_code in ["TIMEOUT", "ERROR", "INVALID"]:
        if log_error(status_code, state, env):
            agent_error = True

    # --- Initialize Moderator and GameState if it's the start of an episode ---
    if not hasattr(env, "moderator") or env.done:  # env.done is true after reset by Kaggle core
        initialize_moderator(state, env)

    moderator: Moderator = env.moderator
    game_state: GameState = env.game_state

    # 1. Collect and parse actions from Kaggle agents
    parsed_player_actions = parse_player_actions(state, moderator, game_state)

    # 2. Advance the Moderator
    moderator.advance(parsed_player_actions)

    # 3. Update Kaggle state (observations, rewards, statuses)
    is_game_done = moderator.is_game_over() or agent_error
    current_info = {}
    if is_game_done:
        record_game_end(state, env, game_state, current_info, agent_error)

    # 4. Moderator interprets player actions, updates game phase, and advance game player actions
    active_player_ids_after_advance = set(moderator.get_active_player_ids())

    # 4.1. Accumulate God mode observations from env for rendering
    global_messages = env.game_state.consume_messages()
    global_data = [rec.serialize() for rec in global_messages]
    env.info[EnvInfoKeys.MODERATOR_OBS].append(global_data)

    # 4.2. Update observations for individual agents
    update_agent_messages(
        state, env, moderator, game_state, is_game_done, current_info, active_player_ids_after_advance, agent_error
    )
    return state


def collect_cost_summary(env) -> CostSummary:
    cost_summary = CostSummary()

    for agent_config in env.configuration.agents:
        player_id = agent_config["id"]
        agent_id = agent_config["agent_id"]

        agent_cost_summary = AgentCostSummary(agent_config=agent_config)

        if isinstance(agents.get(agent_id), AgentFactoryWrapper) and issubclass(
            agents[agent_id].agent_class, LLMWerewolfAgent
        ):
            agent_instance = agents[agent_id].get_instance(player_id)
            if agent_instance:
                cost_tracker = agent_instance.cost_tracker
                agent_cost = AgentCost(
                    total_cost=cost_tracker.query_token_cost.total_costs_usd,
                    prompt_tokens=cost_tracker.prompt_token_cost.total_tokens,
                    completion_tokens=cost_tracker.completion_token_cost.total_tokens,
                )
                agent_cost_summary.costs = agent_cost
                agent_cost_summary.data = cost_tracker

                cost_summary.total_cost += agent_cost.total_cost
                cost_summary.total_prompt_tokens += agent_cost.prompt_tokens
                cost_summary.total_completion_tokens += agent_cost.completion_tokens

        cost_summary.cost_per_agent.append(agent_cost_summary)

    cost_summary.total_tokens = cost_summary.total_prompt_tokens + cost_summary.total_completion_tokens
    return cost_summary


def record_game_end(state, env, game_state, current_info, agent_error):
    # log game end to env.info using GameEndResultsDataEntry
    game_end_entry = next(iter(game_state.get_event_by_name(EventName.GAME_END)), None)
    if game_end_entry and game_end_entry.data:
        current_info.update(game_end_entry.data.model_dump())
    # Record if terminated with agent error. If so, the game record is invalid.
    current_info["terminated_with_agent_error"] = agent_error

    # Record cost from endpoints if any.
    # current_info["cost_summary"] = collect_cost_summary(env).model_dump()

    env.info[EnvInfoKeys.GAME_END] = current_info
    # Determine winner based on game_state.history's GAME_END entry
    if game_end_entry:
        scores = game_end_entry.data.scores
        for i, player_id in enumerate(env.player_id_str_list):
            state[i].reward = scores[player_id]


def update_agent_messages(
    state, env, moderator, game_state, is_game_done, current_info, active_player_ids_after_advance, agent_error
):
    for player_index, player_state in enumerate(state):
        player_id_str = env.player_ids_map[player_index]

        # skip if player not active and game is not done
        if player_id_str not in active_player_ids_after_advance and not is_game_done:
            player_state.status = "INACTIVE"
            continue

        # set the status of active player to ACTIVE
        player_state.status = "ACTIVE"
        player_obj = game_state.get_player_by_id(player_id_str)

        # Observation processing
        new_history_entries = player_obj.consume_messages()

        obs = WerewolfObservationModel(
            player_id=player_obj.id,
            role=player_obj.role.name,
            team=player_obj.role.team.value,
            is_alive=player_obj.alive,
            day=game_state.day_count,
            detailed_phase=moderator.detailed_phase.value,
            all_player_ids=game_state.all_player_ids,
            player_thumbnails=env.player_thumbnails,
            alive_players=[p.id for p in game_state.alive_players()],
            revealed_players=game_state.revealed_players(),
            new_visible_announcements=[entry.description for entry in new_history_entries],
            new_player_event_views=new_history_entries,
            game_state_phase=game_state.phase.value,
        )

        set_raw_observation(player_state, raw_obs=obs)

        # Status
        if is_game_done or agent_error:
            player_state.status = "DONE"
        elif player_id_str in active_player_ids_after_advance:
            player_state.status = "ACTIVE"
        else:
            player_state.status = "INACTIVE"

        # Info
        player_state.info = current_info


def parse_player_actions(state, moderator, game_state):
    parsed_player_actions: Dict[str, Action] = {}
    active_player_ids_from_moderator = moderator.get_active_player_ids()

    for sub_state, player in zip(state, game_state.players):
        player_id_str = player.id
        if player_id_str in active_player_ids_from_moderator and sub_state.status == "ACTIVE":
            serialized_action = sub_state.action
            if serialized_action:
                parsed_player_actions[player_id_str] = create_action(serialized_action)
    return parsed_player_actions


def inject_kaggle_scheduler_info(agents_from_config, env):
    """
    TODO: this is a temporary hack to inject additional info from kaggle scheduler to set up agents. To be removed once
        scheduler has run config generator plugin.
    """
    kaggle_agents_info = env.info.get("Agents")
    if kaggle_agents_info and isinstance(kaggle_agents_info, list):
        for agent, kaggle_agent_info in zip(agents_from_config, kaggle_agents_info):
            display_name = kaggle_agent_info.get("Name", "")
            agent["display_name"] = display_name or agent.get("display_name", "")
            agent["thumbnail"] = kaggle_agent_info.get("ThumbnailUrl", "")


def initialize_moderator(state, env):
    num_players = len(state)

    agents_from_config = env.configuration.agents

    if env.info.get("Agents"):
        inject_kaggle_scheduler_info(agents_from_config, env)

    # below checks for configuration consistency with agent count. If inconsistent, it will cause down stream subtle error.
    if len(agents_from_config) < num_players:
        raise ValueError(
            f"Configuration has {len(agents_from_config)} agents, but {num_players} kaggle agents are present."
        )

    players = create_players_from_agents_config(
        agents_from_config,
        randomize_roles=env.configuration.randomize_roles,
        randomize_ids=env.configuration.randomize_ids,
        seed=env.configuration.seed,
    )

    env.game_state = GameState(
        players=players,
        history={},
        night_elimination_reveal_level=env.configuration.night_elimination_reveal_level,
        day_exile_reveal_level=env.configuration.day_exile_reveal_level,
    )

    env.player_ids_map = {i: p.id for i, p in enumerate(players)}
    env.player_id_str_list = [p.id for p in players]

    env.player_thumbnails = {p.id: p.agent.thumbnail for p in players}
    # Initialize protocols from configuration or defaults
    discussion_protocol = create_protocol(
        env.configuration.get("discussion_protocol", {}), default_name=DEFAULT_DISCUSSION_PROTOCOL_NAME
    )
    day_voting_protocol = create_protocol(
        env.configuration.get("day_voting_protocol", {}), default_name=DEFAULT_VOTING_PROTOCOL_NAME
    )
    night_voting_protocol = create_protocol(
        env.configuration.get("werewolf_night_vote_protocol", {}), default_name=DEFAULT_VOTING_PROTOCOL_NAME
    )

    logger.info(
        f"Interpreter: Using Discussion: {type(discussion_protocol).__name__}, "
        f"Day Voting: {type(day_voting_protocol).__name__}, "
        f"Night WW Voting: {type(night_voting_protocol).__name__}"
    )

    env.moderator = Moderator(
        state=env.game_state,
        discussion=discussion_protocol,
        day_voting=day_voting_protocol,
        night_voting=night_voting_protocol,
        night_elimination_reveal_level=env.configuration.night_elimination_reveal_level,
        day_exile_reveal_level=env.configuration.day_exile_reveal_level,
    )

    env.player_full_visible_history_cache = {p_id: [] for p_id in env.player_id_str_list}
    env.info[EnvInfoKeys.MODERATOR_OBS] = []
    env.agents = agents


def renderer(state, env):
    if not hasattr(env, "moderator") or not hasattr(env, "game_state"):
        return "Game not initialized by interpreter yet."

    game_state: GameState = env.game_state

    lines = []
    for entry in game_state.consume_messages():
        lines.append(entry.description)
    return "\n\n".join(lines)


def html_renderer():
    # TODO: fully remove the need for this empty function in a future cleanup pass.
    pass


jsonpath = path.abspath(path.join(path.dirname(__file__), "werewolf.json"))
with open(jsonpath) as handle:
    specification = json.load(handle)
