from flax import struct
import jax

MAP_TYPES = ["dev0", "random"]


@struct.dataclass
class EnvParams:
    max_steps_in_match: int = 100
    map_type: int = 1
    """Map generation algorithm. Can change between games"""
    map_width: int = 24
    map_height: int = 24
    num_teams: int = 2
    match_count_per_episode: int = 5
    """number of matches to play in one episode"""

    # configs for units
    max_units: int = 16
    init_unit_energy: int = 100
    min_unit_energy: int = 0
    max_unit_energy: int = 400
    unit_move_cost: int = 2
    spawn_rate: int = 3

    unit_sap_cost: int = 10
    """
    The unit sap cost is the amount of energy a unit uses when it saps another unit. Can change between games.
    """
    unit_sap_range: int = 4
    """
    The unit sap range is the range of the unit's sap action.
    """
    unit_sap_dropoff_factor: float = 0.5
    """
    The unit sap dropoff factor multiplied by unit_sap_drain
    """
    unit_energy_void_factor: float = 0.125
    """
    The unit energy void factor multiplied by unit_energy
    """

    # configs for energy nodes
    max_energy_nodes: int = 6
    max_energy_per_tile: int = 20
    min_energy_per_tile: int = -20

    max_relic_nodes: int = 6
    relic_config_size: int = 5
    fog_of_war: bool = True
    """
    whether there is fog of war or not
    """
    unit_sensor_range: int = 2
    """
    The unit sensor range is the range of the unit's sensor.
    Units provide "vision power" over tiles in range, equal to manhattan distance to the unit.

    vision power > 0 that team can see the tiles properties
    """

    # nebula tile params
    nebula_tile_vision_reduction: int = 1
    """
    The nebula tile vision reduction is the amount of vision reduction a nebula tile provides.
    A tile can be seen if the vision power over it is > 0.
    """

    nebula_tile_energy_reduction: int = 0
    """amount of energy nebula tiles reduce from a unit"""

    nebula_tile_drift_speed: float = -0.05
    """
    how fast nebula tiles drift in one of the diagonal directions over time. If positive, flows to the top/right, negative flows to bottom/left
    """
    # TODO (stao): allow other kinds of symmetric drifts?

    energy_node_drift_speed: int = 0.02
    """
    how fast energy nodes will move around over time
    """
    energy_node_drift_magnitude: int = 5

    # option to change sap configurations


env_params_ranges = dict(
    # map_type=[1],
    unit_move_cost=list(range(1, 6)),
    unit_sensor_range=list(range(2, 5)),
    nebula_tile_vision_reduction=list(range(0, 4)),
    nebula_tile_energy_reduction=[0, 0, 10, 25],
    unit_sap_cost=list(range(30, 51)),
    unit_sap_range=list(range(3, 8)),
    unit_sap_dropoff_factor=[0.25, 0.5, 1],
    unit_energy_void_factor=[0.0625, 0.125, 0.25, 0.375],
    # map randomizations
    nebula_tile_drift_speed=[-0.05, -0.025, 0.025, 0.05],
    energy_node_drift_speed=[0.01, 0.02, 0.03, 0.04, 0.05],
    energy_node_drift_magnitude=list(range(3, 6)),
)
