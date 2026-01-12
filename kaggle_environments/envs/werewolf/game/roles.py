import json
import logging
from collections import Counter, defaultdict, deque
from copy import deepcopy
from functools import partial
from typing import Deque, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .actions import HealAction, InspectAction
from .base import BaseModerator, BasePlayer, BaseRole, EventHandler, PlayerID, on_event
from .consts import EventName, Phase, RevealLevel, RoleConst, Team
from .records import (
    Event,
    PlayerEventView,
    RequestDoctorSaveDataEntry,
    RequestSeerRevealDataEntry,
    SeerInspectResultDataEntry,
)

logger = logging.getLogger(__name__)


class Role(BaseRole):
    model_config = ConfigDict(use_enum_values=True)

    name: RoleConst = Field(..., frozen=True)
    team: Team
    night_priority: int = 100  # lower number acts earlier
    descriptions: str


class Werewolf(Role):
    name: RoleConst = RoleConst.WEREWOLF
    team: Team = Team.WEREWOLVES
    night_priority: int = 2
    descriptions: str = "Each night, collaborates with fellow werewolves to vote on eliminating one player."


class Villager(Role):
    name: RoleConst = RoleConst.VILLAGER
    team: Team = Team.VILLAGERS
    descriptions: str = "No special abilities. Participates in the daily vote to eliminate a suspected werewolf."


class DoctorDescription:
    ALLOW_SELF_SAVE = "Each night, may protect one player from a werewolf attack. Doctor is allowed to save themselves during night time."
    NO_SELF_SAVE = "Each night, may protect one player from a werewolf attack. Doctor is NOT allowed to save themselves during night time."
    NO_CONSECUTIVE_SAVE = " Doctor is NOT allowed to save the same player on consecutive nights."


class DoctorStateKey:
    LAST_SAVED_DAY = "last_saved_day"
    LAST_SAVED_PLAYER_ID = "last_saved_player_id"


class Doctor(Role):
    name: RoleConst = RoleConst.DOCTOR
    team: Team = Team.VILLAGERS
    allow_self_save: bool = False
    allow_consecutive_saves: bool = True
    descriptions: str = ""

    @model_validator(mode="after")
    def set_descriptions_default(self) -> "Doctor":
        if self.descriptions == "":
            if self.allow_self_save:
                self.descriptions = DoctorDescription.ALLOW_SELF_SAVE
            else:
                self.descriptions = DoctorDescription.NO_SELF_SAVE
            if not self.allow_consecutive_saves:
                self.descriptions += DoctorDescription.NO_CONSECUTIVE_SAVE
        return self

    @on_event(EventName.NIGHT_START)
    def on_night_starts(self, me: BasePlayer, moderator: BaseModerator, event: Event):
        if me.alive:
            current_day = moderator.state.day_count
            last_saved_day = me.get_role_state(DoctorStateKey.LAST_SAVED_DAY, default=-1)
            last_saved_player_id = me.get_role_state(DoctorStateKey.LAST_SAVED_PLAYER_ID)

            # Reset consecutive save memory if a night was skipped
            if not self.allow_consecutive_saves and last_saved_day != -1 and current_day > last_saved_day + 1:
                me.set_role_state(DoctorStateKey.LAST_SAVED_PLAYER_ID, None)
                last_saved_player_id = None

            valid_candidates = [p.id for p in moderator.state.alive_players()]

            if not self.allow_self_save:
                valid_candidates = [p_id for p_id in valid_candidates if p_id != me.id]

            prompt = "Wake up Doctor. Who would you like to save? "
            if not self.allow_consecutive_saves and last_saved_player_id:
                valid_candidates = [p_id for p_id in valid_candidates if p_id != last_saved_player_id]
                prompt += f'You cannot save the same player on consecutive nights. Player "{last_saved_player_id}" is not a valid target this night. '

            data_entry = RequestDoctorSaveDataEntry(
                valid_candidates=valid_candidates, action_json_schema=json.dumps(HealAction.schema_for_player())
            )
            prompt += f"The options are {data_entry.valid_candidates}."

            moderator.request_action(
                action_cls=HealAction,
                player_id=me.id,
                prompt=prompt,
                data=data_entry,
                event_name=EventName.HEAL_REQUEST,
            )

    @on_event(EventName.HEAL_ACTION)
    def on_heal_action(self, me: BasePlayer, moderator: BaseModerator, event: Event):
        if not me.alive or event.data.actor_id != me.id:
            return

        action = event.data.action
        if isinstance(action, HealAction):
            if not self.allow_self_save and action.target_id == me.id:
                moderator.state.push_event(
                    description=f'Player "{me.id}", doctor is not allowed to self save. '
                    f"Your target is {action.target_id}, which is your own id.",
                    event_name=EventName.ERROR,
                    public=False,
                    visible_to=[me.id],
                )
                return

            if not self.allow_consecutive_saves and action.target_id == me.get_role_state(
                DoctorStateKey.LAST_SAVED_PLAYER_ID
            ):
                moderator.state.push_event(
                    description=f'Player "{me.id}", you cannot save the same player on consecutive nights. '
                    f'Your target "{action.target_id}" was also saved last night.',
                    event_name=EventName.ERROR,
                    public=False,
                    visible_to=[me.id],
                )
                return

            moderator.record_night_save(me.id, action.target_id)
            me.set_role_state(DoctorStateKey.LAST_SAVED_PLAYER_ID, action.target_id)
            me.set_role_state(DoctorStateKey.LAST_SAVED_DAY, moderator.state.day_count)


class SeerDescription:
    REVEAL_ROLE = "Each night, may inspect one player to learn their true role."
    REVEAL_TEAM = "Each night, may inspect one player's team but not their role."


class Seer(Role):
    name: RoleConst = RoleConst.SEER
    team: Team = Team.VILLAGERS
    descriptions: str = ""
    reveal_level: RevealLevel = RevealLevel.ROLE

    @field_validator("reveal_level")
    @classmethod
    def validate_reveal_level(cls, v):
        if v == RevealLevel.NO_REVEAL:
            raise ValueError(f"Setting reveal_level of Seer as {v}. Seer will become useless.")
        return v

    @model_validator(mode="after")
    def set_descriptions_default(self) -> "Seer":
        if self.descriptions == "":
            if self.reveal_level == RevealLevel.ROLE:
                self.descriptions = SeerDescription.REVEAL_ROLE
            elif self.reveal_level == RevealLevel.TEAM:
                self.descriptions = SeerDescription.REVEAL_TEAM
            else:
                raise ValueError(f"reveal_level {self.reveal_level} not supported.")
        return self

    @on_event(EventName.NIGHT_START)
    def on_night_starts(self, me: BasePlayer, moderator: BaseModerator, event: Event):
        if me.alive:
            data_entry = RequestSeerRevealDataEntry(
                valid_candidates=[p.id for p in moderator.state.alive_players() if p != me],
                action_json_schema=json.dumps(InspectAction.schema_for_player()),
            )
            moderator.request_action(
                action_cls=InspectAction,
                player_id=me.id,
                prompt=f"Wake up Seer. Who would you like to see their true {self.reveal_level}? "
                f"The options are {data_entry.valid_candidates}.",
                data=data_entry,
                event_name=EventName.INSPECT_REQUEST,
            )

    @on_event(EventName.INSPECT_ACTION)
    def on_inspect_action(self, me: BasePlayer, moderator: BaseModerator, event: Event):
        action = event.data.action
        if not me.alive or action.actor_id != me.id:
            return
        actor_id = me.id
        target_player = moderator.state.get_player_by_id(action.target_id)
        if target_player:  # Ensure target exists
            role = None
            team = None
            reveal_text = ""
            if self.reveal_level == RevealLevel.ROLE:
                role = target_player.role.name
                team = target_player.role.team
                reveal_text = f'Their role is a "{target_player.role.name}" in team "{target_player.role.team.value}".'
            elif self.reveal_level == RevealLevel.TEAM:
                team = target_player.role.team
                reveal_text = f"Their team is {team}."

            data = SeerInspectResultDataEntry(actor_id=actor_id, target_id=action.target_id, role=role, team=team)
            moderator.state.push_event(
                description=f'Player "{actor_id}", you inspected {target_player.id}. ' + reveal_text,
                event_name=EventName.INSPECT_RESULT,
                public=False,
                visible_to=[actor_id],
                data=data,
            )
        else:
            moderator.state.push_event(
                description=f'Player "{actor_id}", you inspected player "{action.target_id}",'
                f" but this player could not be found.",
                event_name=EventName.ERROR,
                public=False,
                visible_to=[actor_id],
            )


class LLM(BaseModel):
    model_name: str
    properties: Dict = {}


class Agent(BaseModel):
    id: PlayerID
    """The unique name of the player."""

    agent_id: str
    """Id of the agent. Might not be unique (many players might be using the same underlying agent)."""

    display_name: str = ""
    """Agent name shown in the UI and only visible to spectator but not the players. e.g. Pete (base_harness-gemini-2.5-pro)
    base_harness-gemini-2.5-pro is the display_name while Pete is the id. It maybe different from agent_id, 
    e.g. base_harness_v2-gemini-2.5-pro-0506, to reduce the cognitive load of the spectators.
    """

    role: RoleConst
    role_params: Dict = Field(default_factory=dict)
    """Parameters to the Role constructor"""

    thumbnail: Optional[str] = ""
    agent_harness_name: str = "basic_llm"
    llms: List[LLM] = []

    def get_agent_name(self):
        return f"{self.agent_harness_name}({', '.join([llm.model_name for llm in self.llms])})"


class Player(BasePlayer):
    model_config = ConfigDict(use_enum_values=True)

    id: PlayerID
    """The unique name of the player."""

    agent: Agent
    role: BaseRole
    alive: bool = True
    eliminated_during_day: int = -1
    """game starts at night 0, then day 1, night 1, day 2, ..."""

    eliminated_during_phase: Optional[Phase] = None

    _message_queue: Deque[PlayerEventView] = PrivateAttr(default_factory=deque)
    _role_state: Dict = PrivateAttr(default_factory=dict)

    def set_role_state(self, key, value):
        self._role_state[key] = value

    def get_role_state(self, key, default=None):
        return self._role_state.get(key, default)

    def get_event_handlers(self, moderator: BaseModerator) -> Dict[EventName, List[EventHandler]]:
        handlers = defaultdict(list)
        for event_type, handler in self.role.get_event_handlers().items():
            event_handler = partial(handler, self, moderator)
            handlers[event_type].append(event_handler)
        return handlers

    def update(self, entry: PlayerEventView):
        self._message_queue.append(entry)

    def consume_messages(self) -> List[PlayerEventView]:
        messages = list(self._message_queue)
        self._message_queue.clear()
        return messages

    def eliminate(self, day: int, phase: Phase):
        self.alive = False
        self.eliminated_during_day = day
        self.eliminated_during_phase = phase.value

    def report_elimination(self):
        return {
            "player_id": self.id,
            "eliminated_during_day": self.eliminated_during_day,
            "eliminated_during_phase": self.eliminated_during_phase,
        }


ROLE_CLASS_MAP = {
    RoleConst.WEREWOLF.value: Werewolf,
    RoleConst.DOCTOR.value: Doctor,
    RoleConst.SEER.value: Seer,
    RoleConst.VILLAGER.value: Villager,
}


def get_permutation(items: List, seed: int) -> List:
    """
    Generates a deterministic permutation of a list based on a seed.

    This function implements a Fisher-Yates shuffle using a simple Linear
    Congruential Generator (LCG) for pseudo-random number generation. This
    ensures that the permutation is reproducible across different platforms
    and languages, as it does not depend on Python's built-in 'random' module.

    The LCG parameters (m, a, c) are chosen from the glibc standard for
    broad compatibility.

    Args:
        items: The list of items to be permuted.
        seed: An integer used to initialize the random number generator.

    Returns:
        A new list containing the items in a permuted order.
    """
    # LCG parameters (from glibc)
    m = 2**31
    a = 1103515245
    c = 12345

    # Make a copy to avoid modifying the original list
    shuffled_items = list(items)
    n = len(shuffled_items)

    current_seed = seed

    for i in range(n - 1, 0, -1):
        # Generate a pseudo-random number
        current_seed = (a * current_seed + c) % m

        # Get an index j such that 0 <= j <= i
        j = current_seed % (i + 1)

        # Swap elements
        shuffled_items[i], shuffled_items[j] = shuffled_items[j], shuffled_items[i]

    return shuffled_items


def shuffle_roles(agents_config, seed):
    roles_config = [{"role": agent["role"], "role_params": agent.get("role_params", {})} for agent in agents_config]
    permuted_roles_config = get_permutation(roles_config, seed)
    new_agents_config = deepcopy(agents_config)
    for role, agent in zip(permuted_roles_config, new_agents_config):
        agent["role"] = role["role"]
        agent["role_params"] = role["role_params"]
    return new_agents_config


def shuffle_ids(agents_config, seed):
    ids = [agent["id"] for agent in agents_config]
    permuted_ids = get_permutation(ids, seed)
    new_agents_config = deepcopy(agents_config)
    for player_id, agent in zip(permuted_ids, new_agents_config):
        agent["id"] = player_id
    return new_agents_config


def create_players_from_agents_config(
    agents_config: List[Dict], randomize_roles: bool = False, randomize_ids: bool = False, seed: Optional[int] = None
) -> List[Player]:
    if randomize_roles:
        assert seed is not None
        agents_config = shuffle_roles(agents_config, seed)

    if randomize_ids:
        assert seed is not None
        # Note that we have to use a different seed for shuffle_ids vs shuffle_roles, otherwise the ids and roles
        # arrangement will remain the same. Also, using different seed (even just a simple arithmatic addition),
        # LCG ensures that the sequence of random numbers will be uncorrelated.
        agents_config = shuffle_ids(agents_config, seed + 123)

    # check all agents have unique ids
    agent_ids = [agent_config["id"] for agent_config in agents_config]
    if len(agent_ids) != len(set(agent_ids)):
        counts = Counter(agent_ids)
        duplicates = [item for item, count in counts.items() if count > 1 and item is not None]
        if duplicates:
            raise ValueError(f"Duplicate agent ids found: {', '.join(duplicates)}")
    agents = [Agent(**agent_config) for agent_config in agents_config]
    players = [
        Player(id=agent.id, agent=agent, role=ROLE_CLASS_MAP[agent.role](**agent.role_params)) for agent in agents
    ]
    return players
