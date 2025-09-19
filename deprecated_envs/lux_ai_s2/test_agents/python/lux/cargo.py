from dataclasses import dataclass

@dataclass
class UnitCargo:
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0
