"""
Tile class representing a single tile in the game grid.
"""

from ..constants import BUILDING_MAX_HEALTH, HEADQUARTERS_MAX_HEALTH, PLAYER_COLORS, TILE_COLORS, TOWER_MAX_HEALTH


class Tile:
    """Represents a single tile in the grid with type, player ownership, and team info."""

    def __init__(self, tile_data, x, y):
        """
        Initialize a tile from CSV data.

        Args:
            tile_data: String in format "type" or "type_player" or "type_player_team"
            x: X coordinate in grid
            y: Y coordinate in grid
        """
        # Convert to string and strip whitespace
        tile_str = str(tile_data).strip()

        # Handle NaN, empty, or invalid values
        if tile_str in ["", "nan", "None", "NaN"]:
            tile_str = "o"  # Default to grass

        # Split by underscore
        parts = tile_str.split("_")

        self.type = parts[0].strip()  # Strip whitespace from type too
        self.player = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
        self.team = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else None
        self.x = x
        self.y = y

        # Validate tile type - if invalid, default to grass
        valid_types = ["p", "w", "m", "f", "r", "b", "h", "t", "o"]
        if self.type not in valid_types:
            self.type = "o"

        # Tower/Headquarters/Building-specific properties
        if self.type == "t":
            self.max_health = TOWER_MAX_HEALTH
            self.health = TOWER_MAX_HEALTH
            self.original_player = self.player
            self.regenerating = False
        elif self.type == "h":
            self.max_health = HEADQUARTERS_MAX_HEALTH
            self.health = HEADQUARTERS_MAX_HEALTH
            self.original_player = self.player
            self.regenerating = False
        elif self.type == "b":
            self.max_health = BUILDING_MAX_HEALTH
            self.health = BUILDING_MAX_HEALTH
            self.original_player = self.player
            self.regenerating = False
        else:
            self.max_health = None
            self.health = None
            self.original_player = None
            self.regenerating = False

    def get_color(self):
        """Calculate the final color for this tile based on type and player ownership."""
        base_color = TILE_COLORS.get(self.type, (0, 0, 0))

        # For structures (buildings, HQ, towers), emphasize player color more
        if self.player and self.player in PLAYER_COLORS:
            player_color = PLAYER_COLORS[self.player]

            if self.type in ["h", "b", "t"]:
                # Structures: 70% player color, 30% base color
                return tuple(min(int(base * 0.3 + player * 0.7), 255) for base, player in zip(base_color, player_color))
            else:
                # Regular terrain with owner: 60% base, 40% player
                return tuple(min(int(base * 0.6 + player * 0.4), 255) for base, player in zip(base_color, player_color))

        return base_color

    def is_walkable(self):
        """Check if this tile can be walked on."""
        return self.type != "w" and self.type != "o"  # Water is not walkable

    def is_capturable(self):
        """Check if this tile can be captured."""
        return self.type in ["t", "h", "b"]

    def to_dict(self):
        """Convert tile to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "type": self.type,
            "player": self.player,
            "health": self.health,
            "regenerating": self.regenerating,
        }

    @classmethod
    def from_dict(cls, data):
        """Create tile from dictionary."""
        tile_str = data["type"]
        if data.get("player"):
            tile_str += f"_{data['player']}"

        tile = cls(tile_str, data["x"], data["y"])
        if data.get("health") is not None:
            tile.health = data["health"]
        if data.get("regenerating") is not None:
            tile.regenerating = data["regenerating"]

        return tile
