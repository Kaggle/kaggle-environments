from argparse import Namespace
from dataclasses import dataclass
import dataclasses
from typing import Dict, List


def convert_dict_to_ns(x):
    if isinstance(x, dict):
        for k in x:
            x[k] = convert_dict_to_ns(x)
        return Namespace(x)


@dataclass
class UnitConfig:
    METAL_COST: int = 100
    POWER_COST: int = 500
    CARGO_SPACE: int = 1000
    BATTERY_CAPACITY: int = 1500
    CHARGE: int = 1
    INIT_POWER: int = 50
    MOVE_COST: int = 1
    RUBBLE_MOVEMENT_COST: int = 1
    DIG_COST: int = 5
    DIG_RUBBLE_REMOVED: int = 1
    DIG_RESOURCE_GAIN: int = 2
    DIG_LICHEN_REMOVED: int = 10
    SELF_DESTRUCT_COST: int = 10
    RUBBLE_AFTER_DESTRUCTION: int = 1



@dataclass
class EnvConfig:
    ## various options that can be configured if needed


    ### Variable parameters that don't affect game logic much ###
    max_episode_length: int = 1000
    map_size: int = 48
    verbose: int = 1
    
    # this can be disabled to improve env FPS but assume your actions are well formatted
    # During online competition this is set to True
    validate_action_space: bool = True

    ### Constants ###
    # you can only ever transfer in/out 1000 as this is the max cargo space.
    max_transfer_amount: int = 10000
    MIN_FACTORIES: int = 2
    MAX_FACTORIES: int = 5
    CYCLE_LENGTH: int = 50
    DAY_LENGTH: int = 30
    UNIT_ACTION_QUEUE_SIZE: int = 20 # when set to 1, then no action queue is used
    UNIT_ACTION_QUEUE_POWER_COST: Dict[str, int] = dataclasses.field(default_factory=lambda: dict(
        LIGHT=1,
        HEAVY=10
    ))


    MAX_RUBBLE: int = 100
    FACTORY_RUBBLE_AFTER_DESTRUCTION: int = 50
    INIT_WATER_METAL_PER_FACTORY: int = 100 # amount of water and metal units given to each factory
    INIT_POWER_PER_FACTORY: int = 100

    #### LICHEN ####
    MIN_LICHEN_TO_SPREAD: int = 1
    LICHEN_LOST_WITHOUT_WATER: int = 1
    LICHEN_GAINED_WITH_WATER: int = 1
    # cost of watering with a factory is `ceil(# of connected lichen tiles) / (this factor) + 1`
    LICHEN_WATERING_COST_FACTOR: int = 10

    #### Bidding System ####
    BIDDING_SYSTEM: bool = True

    #### Factories ####
    FACTORY_PROCESSING_RATE_WATER: int = 100
    ICE_WATER_RATIO: int = 5
    FACTORY_PROCESSING_RATE_METAL: int = 50
    ORE_METAL_RATIO: int = 5
    # game design note: Factories close to resource cluster = more resources are refined per turn
    # Then the high ice:water and ore:metal ratios encourages transfer of refined resources between 
    # factories dedicated to mining particular clusters which is more possible as it is more compact
    # 


    FACTORY_CHARGE: int = 50
    FACTORY_WATER_CONSUMPTION: int = 1


    #### Units ####
    ROBOTS: Dict[str, UnitConfig] = dataclasses.field(
        default_factory=lambda: dict(
        LIGHT=UnitConfig(
            METAL_COST=10, POWER_COST=50, INIT_POWER=50, CARGO_SPACE=100, BATTERY_CAPACITY=150, CHARGE=1, MOVE_COST=1, RUBBLE_MOVEMENT_COST=1,
            DIG_COST=5,
            SELF_DESTRUCT_COST=5,
            DIG_RUBBLE_REMOVED=1,
            DIG_RESOURCE_GAIN=2,
            DIG_LICHEN_REMOVED=10,
            RUBBLE_AFTER_DESTRUCTION=1,
        ),

        HEAVY=UnitConfig(
            METAL_COST=100, POWER_COST=500, INIT_POWER=500, CARGO_SPACE=1000, BATTERY_CAPACITY=1500, CHARGE=10, MOVE_COST=20, RUBBLE_MOVEMENT_COST=1,
            DIG_COST=100,
            SELF_DESTRUCT_COST=100,
            DIG_RUBBLE_REMOVED=10,
            DIG_RESOURCE_GAIN=20,
            DIG_LICHEN_REMOVED=100,
            RUBBLE_AFTER_DESTRUCTION=10,
        ),
        )
    )

    #### Weather ####
    WEATHER_ID_TO_NAME: list = dataclasses.field(default_factory=lambda : ["NONE","MARS_QUAKE","COLD_SNAP","DUST_STORM","SOLAR_FLARE"])
    NUM_WEATHER_EVENTS_RANGE: List[int] = dataclasses.field(default_factory=lambda:[3,5])
    WEATHER: dict = dataclasses.field(default_factory=lambda:dict(
        MARS_QUAKE=dict(
            # amount of rubble generated under each robot per turn
            RUBBLE=dict(LIGHT=1, HEAVY=10),
            TIME_RANGE=[1, 5]
        ),
        COLD_SNAP=dict(
            # power multiplier required per robot action. 2 -> requires 2x as much power to execute the same action
            POWER_CONSUMPTION=2,
            TIME_RANGE=[10, 30]
        ),
        DUST_STORM=dict(
            # power gain multiplier. .5 -> gain .5x as much power per turn
            POWER_GAIN=0.5,
            TIME_RANGE=[10, 30]
        ),
        SOLAR_FLARE=dict(
            # power gain multiplier. 2 -> gain 2x as much power per turn
            POWER_GAIN=2,
            TIME_RANGE=[10, 30]
        ),
    ))

    @classmethod
    def from_dict(cls, data):
        data["ROBOTS"]["LIGHT"] = UnitConfig(**data["ROBOTS"]["LIGHT"])
        data["ROBOTS"]["HEAVY"] = UnitConfig(**data["ROBOTS"]["HEAVY"])
        return cls(**data)