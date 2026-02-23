"""
Fog of War visibility system for Reinforce Tactics.

This module provides visibility tracking and calculation for each player,
implementing a simple radius-based visibility model (Option A).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .game_state import GameState
    from .unit import Unit


# Visibility state constants
UNEXPLORED = 0  # Never seen - terrain unknown
SHROUDED = 1  # Previously explored - terrain known, units/ownership hidden
VISIBLE = 2  # Currently visible - full information


# Default vision ranges for units (Chebyshev distance)
UNIT_VISION_RANGES: Dict[str, int] = {
    "W": 3,  # Warrior - standard vision
    "M": 3,  # Mage - standard vision
    "C": 3,  # Cleric - standard vision
    "A": 4,  # Archer - extended vision (scout)
    "K": 3,  # Knight - standard vision
    "R": 4,  # Rogue - extended vision (scout)
    "S": 3,  # Sorcerer - standard vision
    "B": 2,  # Barbarian - limited vision
}

# Vision ranges for structures
STRUCTURE_VISION_RANGES: Dict[str, int] = {
    "h": 4,  # Headquarters - large vision radius
    "b": 3,  # Building - standard vision
    "t": 5,  # Tower - best vision (elevated position)
}


@dataclass
class UnitSnapshot:
    """Snapshot of unit information when last visible.

    Used to remember enemy unit positions after they leave vision.
    """

    unit_type: str
    owner: int
    health: int
    max_health: int
    position: Tuple[int, int]
    turn_seen: int


@dataclass
class StructureSnapshot:
    """Snapshot of structure information when last visible."""

    tile_type: str
    owner: Optional[int]
    health: int
    position: Tuple[int, int]
    turn_seen: int


class VisibilityMap:
    """Tracks visibility state for a single player.

    Maintains three layers of information:
    1. Visibility state (unexplored/shrouded/visible) for each tile
    2. Memory of last-seen enemy units
    3. Memory of last-seen structure states

    Args:
        width: Grid width
        height: Grid height
        player: The player this visibility map belongs to
    """

    def __init__(self, width: int, height: int, player: int):
        self.width = width
        self.height = height
        self.player = player

        # Visibility state: 0=unexplored, 1=shrouded, 2=visible
        self.state = np.zeros((height, width), dtype=np.uint8)

        # Memory of last-seen enemy units (position -> UnitSnapshot)
        self.last_seen_units: Dict[Tuple[int, int], UnitSnapshot] = {}

        # Memory of last-seen structures (position -> StructureSnapshot)
        self.last_seen_structures: Dict[Tuple[int, int], StructureSnapshot] = {}

        # Current visibility mask (recomputed each update)
        self._current_visible = np.zeros((height, width), dtype=bool)

    def update(self, game_state: "GameState") -> None:
        """Recalculate visibility based on current unit/structure positions.

        This method:
        1. Marks previously visible tiles as shrouded
        2. Calculates new visibility from all owned units and structures
        3. Updates memory of seen enemy units and structures

        Args:
            game_state: Current game state
        """
        # Step 1: Mark previously visible areas as shrouded (not unexplored)
        self.state[self.state == VISIBLE] = SHROUDED

        # Step 2: Calculate new visibility
        self._current_visible.fill(False)

        # Vision from units (includes terrain bonuses like mountain +1)
        for unit in game_state.units:
            if unit.player == self.player:
                tile = game_state.grid.get_tile(unit.x, unit.y)
                tile_type = tile.type if tile else None
                vision_range = calculate_vision_radius(unit.type, tile_type=tile_type)
                self._add_vision_radius(unit.x, unit.y, vision_range)

        # Vision from structures
        for y in range(self.height):
            for x in range(self.width):
                tile = game_state.grid.get_tile(x, y)
                if tile.player == self.player and tile.type in STRUCTURE_VISION_RANGES:
                    vision_range = calculate_vision_radius(
                        tile.type, is_structure=True)
                    self._add_vision_radius(x, y, vision_range)

        # Step 3: Update state array
        self.state[self._current_visible] = VISIBLE

        # Step 4: Update memory of what we can see
        self._update_memory(game_state)

    def _add_vision_radius(self, cx: int, cy: int, radius: int) -> None:
        """Add circular vision around a point using Chebyshev distance.

        Chebyshev distance (king's movement) creates a square visibility area,
        which is simpler and faster than Euclidean distance circles.

        Args:
            cx: Center x coordinate
            cy: Center y coordinate
            radius: Vision radius in tiles
        """
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Chebyshev distance = max of absolute differences
                    if max(abs(dx), abs(dy)) <= radius:
                        self._current_visible[ny, nx] = True

    def _update_memory(self, game_state: "GameState") -> None:
        """Update memory of seen units and structures.

        Args:
            game_state: Current game state
        """
        turn = game_state.turn_number

        # Clear memory for positions that are now visible (will re-add current info)
        positions_to_clear = [pos for pos in self.last_seen_units if self.is_visible(pos[0], pos[1])]
        for pos in positions_to_clear:
            del self.last_seen_units[pos]

        # Record enemy units we can see
        for unit in game_state.units:
            if unit.player != self.player and self.is_visible(unit.x, unit.y):
                self.last_seen_units[(unit.x, unit.y)] = UnitSnapshot(
                    unit_type=unit.type,
                    owner=unit.player,
                    health=unit.health,
                    max_health=unit.max_health,
                    position=(unit.x, unit.y),
                    turn_seen=turn,
                )

        # Record structures we can see
        for y in range(self.height):
            for x in range(self.width):
                if self.is_visible(x, y):
                    tile = game_state.grid.get_tile(x, y)
                    if tile.type in ("h", "b", "t"):
                        self.last_seen_structures[(x, y)] = StructureSnapshot(
                            tile_type=tile.type, owner=tile.player, health=tile.health, position=(x, y), turn_seen=turn
                        )

    def is_visible(self, x: int, y: int) -> bool:
        """Check if a tile is currently visible.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if tile is currently visible
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.state[y, x] == VISIBLE
        return False

    def is_explored(self, x: int, y: int) -> bool:
        """Check if a tile has ever been explored.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if tile is explored (visible or shrouded)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.state[y, x] >= SHROUDED
        return False

    def get_visibility_state(self, x: int, y: int) -> int:
        """Get the visibility state of a tile.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            UNEXPLORED (0), SHROUDED (1), or VISIBLE (2)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.state[y, x])
        return UNEXPLORED

    def get_visible_mask(self) -> np.ndarray:
        """Get a boolean mask of currently visible tiles.

        Returns:
            2D numpy array where True = visible
        """
        return self.state == VISIBLE

    def get_explored_mask(self) -> np.ndarray:
        """Get a boolean mask of explored tiles.

        Returns:
            2D numpy array where True = explored (visible or shrouded)
        """
        return self.state >= SHROUDED

    def get_last_seen_unit(self, x: int, y: int) -> Optional[UnitSnapshot]:
        """Get the last-seen unit at a position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            UnitSnapshot if a unit was seen there, None otherwise
        """
        return self.last_seen_units.get((x, y))

    def get_last_seen_structure(self, x: int, y: int) -> Optional[StructureSnapshot]:
        """Get the last-seen structure info at a position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            StructureSnapshot if a structure was seen there, None otherwise
        """
        return self.last_seen_structures.get((x, y))

    def clear_stale_unit_memory(self, max_turns: int, current_turn: int) -> None:
        """Remove unit memories older than max_turns.

        Args:
            max_turns: Maximum number of turns to remember units
            current_turn: Current game turn
        """
        stale_positions = [
            pos for pos, snapshot in self.last_seen_units.items() if current_turn - snapshot.turn_seen > max_turns
        ]
        for pos in stale_positions:
            del self.last_seen_units[pos]

    def to_numpy(self) -> np.ndarray:
        """Convert visibility state to numpy array.

        Returns:
            2D numpy array with visibility state values (0, 1, or 2)
        """
        return self.state.copy()


def calculate_vision_radius(
    unit_or_structure_type: str, tile_type: Optional[str] = None, is_structure: bool = False
) -> int:
    """Calculate vision radius for a unit or structure.

    Args:
        unit_or_structure_type: Unit type code ('W', 'M', etc.) or structure type ('h', 'b', 't')
        tile_type: The terrain tile the unit is standing on (for bonuses)
        is_structure: True if this is a structure, False if unit

    Returns:
        Vision radius in tiles
    """
    if is_structure:
        base_range = STRUCTURE_VISION_RANGES.get(unit_or_structure_type, 3)
    else:
        base_range = UNIT_VISION_RANGES.get(unit_or_structure_type, 3)

    # Mountain bonus: +1 vision when standing on mountain (units only)
    if not is_structure and tile_type == "m":
        base_range += 1

    return base_range


def get_visible_units(game_state: "GameState", player: int, include_own: bool = True) -> List["Unit"]:
    """Get list of units visible to a player.

    Args:
        game_state: Current game state
        player: Player to get visible units for
        include_own: Whether to include the player's own units

    Returns:
        List of visible units
    """
    if not game_state.fog_of_war:
        # No fog of war - all units visible
        if include_own:
            return list(game_state.units)
        return [u for u in game_state.units if u.player != player]

    visibility_map = game_state.visibility_maps.get(player)
    if visibility_map is None:
        return list(game_state.units) if include_own else []

    visible = []
    for unit in game_state.units:
        if unit.player == player:
            if include_own:
                visible.append(unit)
        elif visibility_map.is_visible(unit.x, unit.y):
            visible.append(unit)

    return visible


def get_visible_tiles_info(game_state: "GameState", player: int) -> List[Tuple[int, int, "any", int]]:
    """Get list of tiles with visibility information.

    Args:
        game_state: Current game state
        player: Player to get visible tiles for

    Returns:
        List of tuples (x, y, tile, visibility_state)
    """
    if not game_state.fog_of_war:
        # No fog of war - all tiles fully visible
        result = []
        for y in range(game_state.grid.height):
            for x in range(game_state.grid.width):
                tile = game_state.grid.get_tile(x, y)
                result.append((x, y, tile, VISIBLE))
        return result

    visibility_map = game_state.visibility_maps.get(player)
    if visibility_map is None:
        return []

    result = []
    for y in range(game_state.grid.height):
        for x in range(game_state.grid.width):
            tile = game_state.grid.get_tile(x, y)
            vis_state = visibility_map.get_visibility_state(x, y)
            result.append((x, y, tile, vis_state))

    return result
