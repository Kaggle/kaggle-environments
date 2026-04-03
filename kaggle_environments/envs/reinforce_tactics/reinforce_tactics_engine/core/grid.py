"""
Grid class for managing the tile map.
"""

import numpy as np

from .tile import Tile


class TileGrid:
    """Manages the grid of tiles."""

    def __init__(self, map_data):
        """
        Initialize the grid from map data.

        Args:
            map_data: 2D array (pandas DataFrame or numpy array) containing tile information
        """
        self.tiles = []
        self.width = map_data.shape[1]
        self.height = map_data.shape[0]

        for y in range(self.height):
            row = []
            for x in range(self.width):
                tile = Tile(map_data.iloc[y, x] if hasattr(map_data, "iloc") else map_data[y, x], x, y)
                row.append(tile)
            self.tiles.append(row)

    def get_tile(self, x, y):
        """Get tile at coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None

    def get_tiles_by_player(self, player):
        """Get all tiles owned by a player."""
        return [tile for row in self.tiles for tile in row if tile.player == player]

    def get_capturable_tiles(self, player=None):
        """Get all capturable tiles, optionally filtered by player."""
        tiles = [tile for row in self.tiles for tile in row if tile.is_capturable()]
        if player is not None:
            tiles = [tile for tile in tiles if tile.player == player]
        return tiles

    def to_numpy(self):
        """
        Convert grid to numpy representation for RL.

        Returns:
            numpy array of shape (height, width, channels) where channels are:
            0: tile_type (encoded as int)
            1: tile_owner (0 for neutral, 1-4 for players)
            2: structure_hp_percentage (0-100)
        """
        result = np.zeros((self.height, self.width, 3), dtype=np.float32)

        tile_type_encoding = {"p": 0, "w": 1, "m": 2, "f": 3, "r": 4, "b": 5, "h": 6, "t": 7}

        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                result[y, x, 0] = tile_type_encoding.get(tile.type, 0)
                result[y, x, 1] = tile.player if tile.player else 0

                if tile.health is not None and tile.max_health is not None:
                    result[y, x, 2] = (tile.health / tile.max_health) * 100
                else:
                    result[y, x, 2] = 0

        return result

    def to_dict(self):
        """Convert grid to dictionary for serialization."""
        return {
            "width": self.width,
            "height": self.height,
            "tiles": [tile.to_dict() for row in self.tiles for tile in row if tile.is_capturable()],
        }

    @classmethod
    def from_dict(cls, data, map_data):
        """Restore grid from dictionary."""
        grid = cls(map_data)

        # Restore tile states
        for tile_data in data.get("tiles", []):
            x, y = tile_data["x"], tile_data["y"]
            if 0 <= x < grid.width and 0 <= y < grid.height:
                tile = grid.tiles[y][x]
                if tile_data.get("player"):
                    tile.player = tile_data["player"]
                if tile_data.get("health") is not None:
                    tile.health = tile_data["health"]
                if tile_data.get("regenerating") is not None:
                    tile.regenerating = tile_data["regenerating"]

        return grid
