import constants
from typing import List
import math
DIRECTIONS = constants.Constants.DIRECTIONS
RESOURCE_TYPES = constants.Constants.RESOURCE_TYPES


class GameMap:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.map: List[List[Cell]] = [None] * height
        for y in range(0, self.height):
            self.map[y] = [None] * width
            for x in range(0, self.width):
                self.map[y][x] = Cell(x, y)

        def get_cell_by_pos(self, pos) -> Cell:
            return self.map[pos.y][pos.x]

        def get_cell(x, y) -> Cell:
            return self.map[y][x]

        def _setResource(self, type, x, y, amount):
            """
            do not use this function, this is for internal tracking of state
            """
            cell = self.get_cell(x, y)
            cell.resource = Resource(type, amount)


class Resource:
    def __init__(self, type: str, amount: int):
        self.type = type
        self.amount = amount


class Cell:
    def __init__(self, x, y):
        self.pos = Position(x, y)
        self.resource: Resource = None
        self.citytile = None
        self.cooldown = 1


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_adjacent(self, pos):
        dx = self.x - pos.x
        dy = self.y - pos.y
        if abs(dx) + abs(dy) > 1:
            return false
        return true

    def equals(self, pos: Position):
        return self.x == pos.x and self.y == pos.y

    def translate(self, direction, units):
        if direction == DIRECTIONS.NORTH:
            return Position(self.x, self.y - units)
        elif direction == DIRECTIONS.EAST:
            return Position(self.x + units, self.y)
        elif direction == DIRECTIONS.SOUTH:
            return Position(self.x, self.y + units)
        elif direction == DIRECTIONS.WEST:
            return Position(self.x - units, self.y)

    def distance_to(self, pos: Position):
        """
        Returns Euclidiean (L2) distance to pos
        """
        dx = pos.x - self.x
        dy = pos.y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def direction_to(self, target_pos: Position):
        """
        Return closest position to target_pos from this position
        """
        checkDirections = [
            DIRECTIONS.NORTH,
            DIRECTIONS.EAST,
            DIRECTIONS.SOUTH,
            DIRECTIONS.WEST,
        ]
        closest_dist = 9999999
        closest_dir = None
        for direction in check_dirs:
            newpos = self.translate(direction, 1)
            dist = target_pos.distance_to(newpos)
            if dist < closest_dist:
                closest_dir = direction
                closest_dist = dist
        return closest_dir
