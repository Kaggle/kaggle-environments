from typing import Literal, List, Any, Tuple, Optional
from enum import Enum, auto
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class Phase(str, Enum):
    DAY = "Day"
    NIGHT = "Night"


class Team(str, Enum):
    VILLAGERS = "Villagers"
    WEREWOLVES = "Werewolves"


class RoleConst(str, Enum):
    VILLAGER = "Villager"
    WEREWOLF = "Werewolf"
    DOCTOR = "Doctor"
    SEER = "Seer"


class Role(BaseModel):
    name: str = Field(..., frozen=True)
    team: Team
    night_priority: int = 100  # lower number acts earlier
    descriptions: str

    def night_action(self, actor, target, state):
        pass
    
    def obs(self, state):
        pass


class Werewolf(Role):
    name: str = RoleConst.WEREWOLF
    team: Team = Team.WEREWOLVES
    night_priority: int = 2

    def night_action(self, actor, target, state):
        state.queue_eliminate_vote(target) # Assuming queue_eliminate_vote will be the new name or similar


class Villager(Role):
    name: str = RoleConst.VILLAGER
    team: Team = Team.VILLAGERS


class Doctor(Role):
    name: str = RoleConst.DOCTOR
    team: Team = Team.VILLAGERS

    def night_action(self, actor, target, state):
        state.queue_save_vote(target)


class Seer(Role):
    name: str = RoleConst.SEER
    team: Team = Team.VILLAGERS

    def night_action(self, actor, target, state):
        state.queue_seer_action(actor, target)


class Player(BaseModel):
    id: str
    role: Role
    alive: bool = True
