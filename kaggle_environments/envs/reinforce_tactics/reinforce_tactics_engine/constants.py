"""
Game constants for the 2D Strategy Game.
"""

from enum import Enum


class TileType(Enum):
    """Enumeration of tile types in the game."""

    GRASS = "p"
    WATER = "w"
    MOUNTAIN = "m"
    FOREST = "f"
    ROAD = "r"
    BUILDING = "b"
    HEADQUARTERS = "h"
    TOWER = "t"
    OCEAN = "o"

    @classmethod
    def from_code(cls, code: str) -> "TileType":
        """Get TileType from single-letter code."""
        for tile_type in cls:
            if tile_type.value == code:
                return tile_type
        raise ValueError(f"Unknown tile code: {code}")

    def is_walkable(self) -> bool:
        """Check if this tile type can be walked on."""
        return self not in (TileType.WATER, TileType.OCEAN)

    def is_capturable(self) -> bool:
        """Check if this tile type can be captured."""
        return self in (TileType.TOWER, TileType.HEADQUARTERS, TileType.BUILDING)


# Display settings
TILE_SIZE = 32
FPS = 60
MIN_MAP_SIZE = 20
MIN_STRIP_SIZE = 6  # Minimum size to preserve when stripping water borders

# Tile type colors (fallback when images aren't available)
# Made more distinct and vibrant
TILE_COLORS = {
    TileType.GRASS.value: (100, 200, 100),  # Grass - Bright green
    TileType.WATER.value: (50, 120, 200),  # Water - Blue
    TileType.MOUNTAIN.value: (150, 150, 150),  # Mountain - Light gray
    TileType.FOREST.value: (34, 139, 34),  # Forest - Forest green
    TileType.ROAD.value: (160, 130, 80),  # Road - Brown/tan
    TileType.BUILDING.value: (180, 180, 180),  # Building - Light gray (player-colored)
    TileType.HEADQUARTERS.value: (200, 200, 50),  # Headquarters - Yellow (player-colored)
    TileType.TOWER.value: (220, 220, 220),  # Tower - Light gray
    TileType.OCEAN.value: (0, 39, 232),  # Ocean - Dark Blue
    # Keep string keys for backwards compatibility
    "p": (100, 200, 100),
    "w": (50, 120, 200),
    "m": (150, 150, 150),
    "f": (34, 139, 34),
    "r": (160, 130, 80),
    "b": (180, 180, 180),
    "h": (200, 200, 50),
    "t": (220, 220, 220),
    "o": (0, 39, 232),
}

# Player colors - Made more vibrant
PLAYER_COLORS = {
    1: (255, 50, 50),  # Red - Brighter
    2: (77, 121, 255),  # Blue - Brighter
    3: (50, 255, 50),  # Green - Brighter
    4: (255, 255, 50),  # Yellow - Brighter
}

# Unit colors
UNIT_COLORS = {
    "W": (139, 69, 19),  # Brown (Warrior)
    "M": (138, 43, 226),  # Purple (Mage)
    "C": (255, 215, 0),  # Gold (Cleric)
    "B": (0, 215, 0),  # Barbarian (Green)
    "A": (34, 139, 34),  # Archer (Forest Green)
    "K": (192, 192, 192),  # Knight (Silver)
    "R": (64, 64, 64),  # Rogue (Dark Gray)
    "S": (0, 191, 255),  # Sorcerer (Deep Sky Blue)
}

# Unit costs and properties
UNIT_DATA = {
    "W": {
        "static_path": "warrior.png",
        "animation_path": "warrior",
        "name": "Warrior",
        "cost": 200,
        "color": (139, 69, 19),
        "movement": 3,
        "health": 15,
        "attack": 10,
        "defence": 6,
    },
    "M": {
        "static_path": "mage.png",
        "animation_path": "mage",
        "name": "Mage",
        "cost": 300,
        "color": (138, 43, 226),
        "movement": 2,
        "health": 10,
        "attack": {"adjacent": 8, "range": 12},
        "defence": 4,
    },
    "C": {
        "static_path": "cleric.png",
        "animation_path": "cleric",
        "name": "Cleric",
        "cost": 200,
        "color": (255, 215, 0),
        "movement": 2,
        "health": 8,
        "attack": 2,
        "defence": 4,
    },
    "B": {
        "static_path": "barbarian.png",
        "animation_path": "barbarian",
        "name": "Barbarian",
        "cost": 400,
        "color": (0, 215, 0),
        "movement": 5,
        "health": 20,
        "attack": 10,
        "defence": 2,
    },
    "A": {
        "static_path": "archer.png",
        "animation_path": "archer",
        "name": "Archer",
        "cost": 250,
        "color": (34, 139, 34),
        "movement": 3,
        "health": 15,
        "attack": 5,
        "defence": 1,
    },
    "K": {
        "static_path": "knight.png",
        "animation_path": "knight",
        "name": "Knight",
        "cost": 350,
        "color": (192, 192, 192),
        "movement": 4,
        "health": 18,
        "attack": 8,
        "defence": 5,
    },
    "R": {
        "static_path": "rogue.png",
        "animation_path": "rogue",
        "name": "Rogue",
        "cost": 350,
        "color": (64, 64, 64),
        "movement": 4,
        "health": 12,
        "attack": 9,
        "defence": 3,
    },
    "S": {
        "static_path": "sorcerer.png",
        "animation_path": "sorcerer",
        "name": "Sorcerer",
        "cost": 400,
        "color": (0, 191, 255),
        "movement": 2,
        "health": 10,
        "attack": {"adjacent": 6, "range": 8},
        "defence": 3,
    },
}

# Starting gold for each player
STARTING_GOLD = 250

# Income rates
HEADQUARTERS_INCOME = 150
BUILDING_INCOME = 100
TOWER_INCOME = 50

# Structure health
TOWER_MAX_HEALTH = 30
BUILDING_MAX_HEALTH = 40
HEADQUARTERS_MAX_HEALTH = 50

# Structure regeneration rate (percentage of max HP per turn)
STRUCTURE_REGEN_RATE = 0.5

# Combat
COUNTER_ATTACK_MULTIPLIER = 0.8
DEFENCE_REDUCTION_PER_POINT = 0.05  # Each defence point reduces damage by 5%

# Special ability bonuses
CHARGE_BONUS = 0.5  # Knight: +50% damage if moved 3+ tiles
CHARGE_MIN_DISTANCE = 3  # Minimum tiles moved to trigger Charge
FLANK_BONUS = 0.5  # Rogue: +50% damage if enemy is adjacent to a friendly unit
ROGUE_EVADE_CHANCE = 0.15  # Rogue: 15% chance to dodge counter-attacks

# Status effects
PARALYZE_DURATION = 3
PARALYZE_COOLDOWN = 2  # Turns before Mage can use Paralyze again
HEAL_AMOUNT = 5
HASTE_COOLDOWN = 3  # Turns before Sorcerer can use Haste again

# Rogue forest bonus
ROGUE_FOREST_EVADE_BONUS = 0.15  # Additional 15% dodge chance when in forest (15% + 15% = 30%)

# Sorcerer buff abilities
SORCERER_BUFF_DURATION = 3  # Turns the buff lasts
SORCERER_BUFF_COOLDOWN = 3  # Turns before Sorcerer can use buff again
SORCERER_DEFENCE_BUFF_AMOUNT = 0.35  # 35% damage reduction
SORCERER_ATTACK_BUFF_AMOUNT = 0.35  # 35% damage increase

# Tile type mapping (string code -> display name)
# Kept for backwards compatibility
TILE_TYPES = {
    TileType.GRASS.value: "GRASS",
    TileType.WATER.value: "WATER",
    TileType.MOUNTAIN.value: "MOUNTAIN",
    TileType.FOREST.value: "FOREST",
    TileType.ROAD.value: "ROAD",
    TileType.BUILDING.value: "BUILDING",
    TileType.HEADQUARTERS.value: "HEADQUARTERS",
    TileType.TOWER.value: "TOWER",
    TileType.OCEAN.value: "OCEAN",
    # Also keep simple string keys for backwards compatibility
    "p": "GRASS",
    "w": "WATER",
    "m": "MOUNTAIN",
    "f": "FOREST",
    "r": "ROAD",
    "b": "BUILDING",
    "h": "HEADQUARTERS",
    "t": "TOWER",
    "o": "OCEAN",
}

# Tile images
TILE_IMAGES = {
    "GRASS": "grass.png",
    "WATER": "water.png",
    "OCEAN": "ocean.png",
    "MOUNTAIN": "mountain.png",
    "FOREST": "forest.png",
    "ROAD": "road.png",
    "TOWER": "city.png",
    "BUILDING": "building.png",
    "HEADQUARTERS": "headquarters.png",
}

# Animation configuration for sprite sheets
# Sprite sheets should be organized with animation states in rows:
# - Row 0: Idle animation
# - Row 1: Move down
# - Row 2: Move up
# - Row 3: Move left
# - Row 4: Move right (optional - can be mirrored from left)
ANIMATION_CONFIG = {
    # Default frame dimensions
    "frame_width": 32,
    "frame_height": 32,
    # Row index to animation state mapping
    "state_rows": {
        0: "idle",
        1: "move_down",
        2: "move_up",
        3: "move_left",
        4: "move_right",
    },
    # Animation state configurations
    "states": {
        "idle": {
            "frames": 4,  # Number of frames in animation
            "speed": 0.2,  # Seconds per frame (slower for idle)
        },
        "move_down": {
            "frames": 4,
            "speed": 0.1,  # Faster for movement
        },
        "move_up": {
            "frames": 4,
            "speed": 0.1,
        },
        "move_left": {
            "frames": 4,
            "speed": 0.1,
        },
        "move_right": {
            "frames": 4,
            "speed": 0.1,
        },
    },
    # Per-unit type overrides (optional)
    # Example: 'units': {'W': {'frame_width': 48, 'frame_height': 48}}
    "units": {},
}
