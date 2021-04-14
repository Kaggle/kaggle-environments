from .game_map import Position
from .constants import Constants
from typing import Dict
from .game_constants import GAME_CONSTANTS
UNIT_TYPES = Constants.UNIT_TYPES
class Player():
    def __init__(self,team):
        self.team = team
        self.research_points = 0
        self.units = []
        self.cities: Dict[str, City] = {}
    def researched_coal(self):
        return self.researchPoints >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"]
    def researched_uanium(self):
        return self.researchPoints >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]

class City:
    def __init__(self, teamid, cityid, fuel, light_upkeep):
        self.cityid = cityid;
        self.team = teamid;
        self.fuel = fuel;
        self.citytiles = [];
        self.light_upkeep = light_upkeep;
    def add_city_tile(self, x, y, cooldown):
        ct = CityTile(self.team, self.cityid, x, y, cooldown)
        self.citytiles.append(ct)
        return ct;
    def get_light_upkeep(self):
        return self.light_upkeep

class CityTile:
    def __init__(self, teamid, cityid, x, y, cooldown):
        self.cityid = cityid
        self.team = teamid
        self.pos = Position(x, y)
        self.cooldown = cooldown
    def can_act(self):
        """
        Whether or not this unit can research or build
        """
        return self.cooldown == 0
    def research(self):
        """
        returns command to ask this tile to research this turn
        """
        return "r {} {}".format(self.pos.x, self.pos.y)
    def build_worker(self):
        """
        returns command to ask this tile to build a worker this turn
        """
        return "bw {} {}".format(self.pos.x, self.pos.y)
#   /** returns command to ask this tile to build a cart this turn */
    def build_cart(self):
        """
        returns command to ask this tile to build a cart this turn
        """
        return "bc {} {}".format(self.pos.x, self.pos.y)

class Cargo:
    def __init__(self):
        self.wood = 0
        self.coal = 0
        self.uranium = 0

class Unit:
    def __init__(self, teamid, u_type, unitid, x, y, cooldown, wood, coal, uranium):
        self.pos = Position(x, y);
        self.team = teamid;
        self.id = unitid;
        self.type = u_type;
        self.cooldown = cooldown;
        self.cargo = Cargo()
        self.cargo.wood = wood
        self.cargo.coal = coal
        self.cargo.uranium = uranium
    def is_worker(self):
        return self.type == UNIT_TYPES.WORKER

    def is_cart(self):
        return self.type == UNIT_TYPES.CART

    def getCargoSpaceLeft(self):
        """
        get cargo space left in this unit
        """
        spaceused = self.cargo.wood + self.cargo.coal + self.cargo.uranium;
        if self.type == UNIT_TYPES.WORKER:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] - spaceused;
        else:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] - spaceused;
    
    def can_move(self):
        """
        whether or not the unit can move or not
        """
        return self.cooldown == 0

    def move(self, dir):
        """
        return the command to move unit in the given direction
        """
        return "m {} {}".format(self.id, dir)

    def transfer(self, src_id, dest_id, resourceType, amount):
        """
        return the command to transfer a resource from a source unit to a destination unit as specified by their ids
        """
        return "t {} {} {} {}".format(src_id, dest_id, resourceType, amount)

    def build_city(self):
        """
        return the command to build a city right under the worker
        """
        return "bcity {}".format(self.id)

    def pillage(self):
        """
        return the command to pillage whatever is underneath the worker
        """
        return "p {}".format(self.id)