from dataclasses import dataclass
from enum import Enum
from typing import List
try:
    from typing import TypedDict    
except:
    from typing_extensions import TypedDict

from luxai_s2.config import EnvConfig
from luxai_s2.globals import TERM_COLORS

try:
    from termcolor import colored
except:
    pass


@dataclass
class FactionInfo:
    color: str = "none"
    alt_color: str = "red"
    faction_id: int = -1


class FactionTypes(Enum):
    AlphaStrike = FactionInfo(color="yellow", faction_id=0)
    MotherMars = FactionInfo(color="green", faction_id=1)
    TheBuilders = FactionInfo(color="blue", faction_id=2)
    FirstMars = FactionInfo(color="red", faction_id=3)


class TeamStateDict(TypedDict):
    team_id: int
    faction: str
    water: int
    metal: int
    factories_to_place: int
    factory_strains: List[int]
    place_first: bool
    bid: int


class Team:
    def __init__(self, team_id: int, agent: str, faction: FactionTypes = None) -> None:
        self.faction = faction
        self.team_id = team_id
        # the key used to differentiate ownership of things in state
        self.agent = agent

        self.init_water = 0
        self.init_metal = 0
        self.factories_to_place = 0
        self.factory_strains = []

        # whether this team gets to place factories down first or not. The bid winner has this set to True.
        # If tied, player_0's team has this True
        self.place_first = False

        self.bid = 0

    def state_dict(self) -> TeamStateDict:
        return dict(
            team_id=self.team_id,
            faction=self.faction.name,
            # note for optimization, water,metal, factories_to_place doesn't change after the early game.
            water=self.init_water,
            metal=self.init_metal,
            factories_to_place=self.factories_to_place,
            factory_strains=self.factory_strains,
            place_first=self.place_first,
            bid=self.bid,
        )

    def __str__(self) -> str:
        out = f"[Player {self.team_id}]"
        if TERM_COLORS:
            return colored(out, self.faction.value.color)
        return out
