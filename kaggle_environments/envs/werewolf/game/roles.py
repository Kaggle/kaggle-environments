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


def create_players_from_roles_and_ids(role_strings: List[str], player_ids: List[str]) -> List[Player]:
    """
    Initializes a list of Player instances given a list of role strings and player IDs.

    Args:
        role_strings: A list of strings representing player roles (e.g., "Werewolf", "Doctor").
                      These strings should match the values in RoleConst enum.
        player_ids: A list of unique strings representing player IDs.

    Returns:
        A list of Player instances.

    Raises:
        ValueError: If the lengths of role_strings and player_ids do not match,
                    or if an unknown role string is encountered.
    """
    if len(role_strings) != len(player_ids):
        raise ValueError("The number of roles must match the number of player IDs.")

    # Mapping from RoleConst string value to the actual Role class constructor
    role_class_map = {
        RoleConst.WEREWOLF.value: Werewolf,
        RoleConst.DOCTOR.value: Doctor,
        RoleConst.SEER.value: Seer,
        RoleConst.VILLAGER.value: Villager,
    }

    players: List[Player] = []
    for role_str, player_id in zip(role_strings, player_ids):
        role_class: type[Role] = role_class_map.get(role_str)
        if role_class is None:
            raise ValueError(f"Unknown role string: '{role_str}'. Must be one of {list(role_class_map.keys())}")
        
        players.append(Player(id=player_id, role=role_class()))

    return players