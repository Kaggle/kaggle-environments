from typing import List, Deque, Optional, Dict
import logging
from collections import deque

from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from .consts import Team, RoleConst, Phase
from .records import HistoryEntry

logger = logging.getLogger(__name__)


class Role(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: RoleConst = Field(..., frozen=True)
    team: Team
    night_priority: int = 100  # lower number acts earlier
    descriptions: str

    def night_action(self, actor, target, state):
        pass
    
    def obs(self, state):
        pass


class Werewolf(Role):
    name: RoleConst = RoleConst.WEREWOLF
    team: Team = Team.WEREWOLVES
    night_priority: int = 2
    descriptions: str = "A member of the Werewolf team. At night, works with other werewolves to vote on eliminating one player."

    def night_action(self, actor, target, state):
        state.queue_eliminate_vote(target) # Assuming queue_eliminate_vote will be the new name or similar


class Villager(Role):
    name: RoleConst = RoleConst.VILLAGER
    team: Team = Team.VILLAGERS
    descriptions: str = "A member of the Villagers team. Has no special abilities other than their vote during the day."


class Doctor(Role):
    name: RoleConst = RoleConst.DOCTOR
    team: Team = Team.VILLAGERS
    descriptions: str = "A member of the Villagers team. Each night, can choose one player to protect from a werewolf attack."

    def night_action(self, actor, target, state):
        state.queue_save_vote(target)


class Seer(Role):
    name: RoleConst = RoleConst.SEER
    team: Team = Team.VILLAGERS
    descriptions: str = "A member of the Villagers team. Each night, can choose one player to inspect and learn their true role."

    def night_action(self, actor, target, state):
        state.queue_seer_action(actor, target)


class LLM(BaseModel):
    model_name: str
    properties: Dict = {}


class Agent(BaseModel):
    id: str
    """The unique name of the player."""

    agent_id: str
    """Id of the agent. Might not be unique (many players might be using the same underlying agent)."""

    role: str
    thumbnail: Optional[str] = None
    agent_harness_name: str = "basic_llm"
    llms: List[LLM] = []

    def get_agent_name(self):
        return f"{self.agent_harness_name}({', '.join([llm.model_name for llm in self.llms])})"


class Player(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    """The unique name of the player."""

    agent: Agent
    role: Role
    alive: bool = True
    eliminated_during_day: int = -1
    """game starts at night 0, then day 1, night 1, day 2, ..."""

    eliminated_during_phase: Optional[str] = None

    _message_queue: Deque[HistoryEntry] = PrivateAttr(default_factory=deque)

    def update(self, entry: HistoryEntry):
        self._message_queue.append(entry)
    
    def consume_messages(self) -> List[HistoryEntry]:
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
            "eliminated_during_phase": self.eliminated_during_phase
        }


ROLE_CLASS_MAP = {
    RoleConst.WEREWOLF.value: Werewolf,
    RoleConst.DOCTOR.value: Doctor,
    RoleConst.SEER.value: Seer,
    RoleConst.VILLAGER.value: Villager,
}


def create_players_from_agents_config(agents_config: List[Dict]) -> List[Player]:
    agents = [Agent(**agent_config) for agent_config in agents_config]
    players = [Player(id=agent.id, agent=agent, role=ROLE_CLASS_MAP[agent.role]()) for agent in agents]
    return players
