import functools
import chex
import flax
import jax
import jax.numpy as jnp
from flax import struct

from luxai_s3.params import MAP_TYPES, EnvParams
from luxai_s3.utils import to_numpy
EMPTY_TILE = 0
NEBULA_TILE = 1
ASTEROID_TILE = 2

ENERGY_NODE_FNS = [
    lambda d, x, y, z: jnp.sin(d * x + y) * z, lambda d, x, y, z: (x / (d + 1) + y) * z
]

@struct.dataclass
class UnitState:
    position: chex.Array
    """Position of the unit with shape (2) for x, y"""
    energy: int
    """Energy of the unit"""

@struct.dataclass
class MapTile:
    energy: int
    """Energy of the tile, generated via energy_nodes and energy_node_fns"""
    tile_type: int
    """Type of the tile"""

@struct.dataclass
class EnvState:
    units: UnitState
    """Units in the environment with shape (T, N, 3) for T teams, N max units, and 3 features.

    3 features are for position (x, y), and energy
    """
    units_mask: chex.Array
    """Mask of units in the environment with shape (T, N) for T teams, N max units"""
    energy_nodes: chex.Array
    """Energy nodes in the environment with shape (N, 2) for N max energy nodes, and 2 features.

    2 features are for position (x, y)
    """
    
    energy_node_fns: chex.Array
    """Energy node functions for computing the energy field of the map. They describe the function with a sequence of numbers
    
    The first number is the function used. The subsequent numbers parameterize the function. The function is applied to distance of map tile to energy node and the function parameters.
    """

    # energy_field: chex.Array
    # """Energy field in the environment with shape (H, W) for H height, W width. This is generated from other state"""
    
    energy_nodes_mask: chex.Array
    """Mask of energy nodes in the environment with shape (N) for N max energy nodes"""
    relic_nodes: chex.Array
    """Relic nodes in the environment with shape (N, 2) for N max relic nodes, and 2 features.

    2 features are for position (x, y)
    """
    relic_node_configs: chex.Array
    """Relic node configs in the environment with shape (N, K, K) for N max relic nodes and a KxK relic configuration"""
    relic_nodes_mask: chex.Array
    """Mask of relic nodes in the environment with shape (T, N) for T teams, N max relic nodes"""
    relic_nodes_map_weights: chex.Array
    """Map of relic nodes in the environment with shape (H, W) for H height, W width. True if a relic node is present, False otherwise. This is generated from other state"""

    map_features: MapTile
    """Map features in the environment with shape (W, H, 2) for W width, H height
    """

    sensor_mask: chex.Array
    """Sensor mask in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    vision_power_map: chex.Array
    """Vision power map in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""

    steps: int = 0
    """steps taken in the environment"""
    match_steps: int = 0
    """steps taken in the current match"""

    def get_obs(self):
        return self
    
@struct.dataclass
class EnvObs:
    """Partial observation of environment"""
    units: UnitState
    """Units in the environment with shape (T, N, 3) for T teams, N max units, and 3 features.

    3 features are for position (x, y), and energy
    """
    units_mask: chex.Array
    """Mask of units in the environment with shape (T, N) for T teams, N max units"""
    
    sensor_mask: chex.Array
    
    map_features: MapTile
    """Map features in the environment with shape (W, H, 2) for W width, H height
    """
    
    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""
    steps: int = 0
    """steps taken in the environment"""
    match_steps: int = 0
    """steps taken in the current match"""
    

def serialize_env_states(env_states: list[EnvState]):
    def serialize_array(root: EnvState, arr, key_path: str = ""):
        if key_path in ["vision_power_map", "relic_nodes_mask", "energy_node_fns", "relic_nodes_map_weights"]:
            return None
        if key_path == "relic_nodes":
            return root.relic_nodes[root.relic_nodes_mask].tolist()
        if key_path == "energy_nodes":
            return root.energy_nodes[root.energy_nodes_mask].tolist()
        if isinstance(arr, jnp.ndarray):
            return arr.tolist()
        elif isinstance(arr, dict):
            ret = dict()
            for k, v in arr.items():
                new_key = key_path + "/" + k if key_path else k
                new_val = serialize_array(root, v, new_key)
                if new_val is not None:
                    ret[k] = new_val
            return ret
# TODO (stao): to optimize save file we can store deltas of map info instead. might not be able to do this with kaggle replays though.
        return arr
    steps = []
    for state in env_states:
        state_dict = flax.serialization.to_state_dict(state)
        steps.append(serialize_array(state, state_dict))

    return steps

def serialize_env_actions(env_actions: list):
    def serialize_array(arr, key_path: str = ""):
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        elif isinstance(arr, jnp.ndarray):
            return arr.tolist()
        elif isinstance(arr, dict):
            ret = dict()
            for k, v in arr.items():
                new_key = key_path + "/" + k if key_path else k
                new_val = serialize_array(v, new_key)
                if new_val is not None:
                    ret[k] = new_val
            return ret

        return arr
    steps = []
    for state in env_actions:
        state = flax.serialization.to_state_dict(state)
        steps.append(serialize_array(state))

    return steps


# @struct.dataclass
# class EnvObs:
#     """Observation of the environment. A subset of the environment state due to partial observability."""

#     units: chex.Array
#     units_mask: chex.Array
#     """Mask of units in the environment with shape (T, N) for T teams, N max units"""


def state_to_flat_obs(state: EnvState) -> chex.Array:
    pass


def flat_obs_to_state(flat_obs: chex.Array) -> EnvState:
    pass


def gen_state(key: chex.PRNGKey, params: EnvParams) -> EnvState:
    generated = gen_map(key, params)
    relic_nodes_map_weights = jnp.zeros(
        shape=(params.map_width, params.map_height), dtype=jnp.int16
    )

    # TODO (this could be optimized better)
    def update_relic_node(relic_nodes_map_weights, relic_data):
        relic_node, relic_node_config, mask = relic_data
        start_y = relic_node[1] - params.relic_config_size // 2
        start_x = relic_node[0] - params.relic_config_size // 2
        for dy in range(params.relic_config_size):
            for dx in range(params.relic_config_size):
                y, x = start_y + dy, start_x + dx
                valid_pos = jnp.logical_and(
                    jnp.logical_and(y >= 0, x >= 0),
                    jnp.logical_and(y < params.map_height, x < params.map_width),
                )
                relic_nodes_map_weights = jnp.where(
                    valid_pos & mask,
                    relic_nodes_map_weights.at[x, y].add(relic_node_config[dy, dx]),
                    relic_nodes_map_weights,
                )
        return relic_nodes_map_weights, None

    # this is really slow...
    relic_nodes_map_weights, _ = jax.lax.scan(
        update_relic_node,
        relic_nodes_map_weights,
        (
            generated["relic_nodes"],
            generated["relic_node_configs"],
            generated["relic_nodes_mask"],
        ),
    )
    state = EnvState(
        units=UnitState(position=jnp.zeros(shape=(params.num_teams, params.max_units, 2), dtype=jnp.int16), energy=jnp.zeros(shape=(params.num_teams, params.max_units, 1), dtype=jnp.int16)),
        units_mask=jnp.zeros(
            shape=(params.num_teams, params.max_units), dtype=jnp.bool
        ),
        team_points=jnp.zeros(shape=(params.num_teams), dtype=jnp.int32),
        energy_nodes=generated["energy_nodes"],
        energy_node_fns=generated["energy_node_fns"],
        energy_nodes_mask=generated["energy_nodes_mask"],
        # energy_field=jnp.zeros(shape=(params.map_height, params.map_width), dtype=jnp.int16),
        relic_nodes=generated["relic_nodes"],
        relic_nodes_mask=generated["relic_nodes_mask"],
        relic_node_configs=generated["relic_node_configs"],
        relic_nodes_map_weights=relic_nodes_map_weights,
        sensor_mask=jnp.zeros(
            shape=(params.num_teams, params.map_height, params.map_width),
            dtype=jnp.bool,
        ),
        vision_power_map=jnp.zeros(shape=(params.num_teams, params.map_height, params.map_width), dtype=jnp.int16),
        map_features=generated["map_features"],
    )

    # state = spawn_unit(state, 0, 0, [0, 0], params)
    # state = spawn_unit(state, 0, 1, [0, 0], params)
    # state = spawn_unit(state, 0, 2, [0, 0])
    # state = spawn_unit(state, 1, 0, [15, 15], params)
    # state = spawn_unit(state, 1, 1, [15, 15], params)
    # state = spawn_unit(state, 1, 2, [15, 15])
    return state


def spawn_unit(
    state: EnvState, team: int, unit_id: int, position: chex.Array, params: EnvParams
) -> EnvState:
    unit_state = state.units
    unit_state = unit_state.replace(position=unit_state.position.at[team, unit_id, :].set(jnp.array(position, dtype=jnp.int16)))
    unit_state = unit_state.replace(energy=unit_state.energy.at[team, unit_id, :].set(jnp.array([params.init_unit_energy], dtype=jnp.int16)))
    # state = state.replace(
    #     units
    #     # units=state.units.at[team, unit_id, :].set(
    #     #     jnp.array([position[0], position[1], 0], dtype=jnp.int16)
    #     # )
    # )
    state = state.replace(units=unit_state, units_mask=state.units_mask.at[team, unit_id].set(True))
    return state

def set_tile(map_features: MapTile, x: int, y: int, tile_type: int) -> MapTile:
    return map_features.replace(tile_type=map_features.tile_type.at[x, y].set(tile_type))


def gen_map(key: chex.PRNGKey, params: EnvParams) -> chex.Array:
    map_features = MapTile(energy=jnp.zeros(
        shape=(params.map_height, params.map_width), dtype=jnp.int16
    ), tile_type=jnp.zeros(
        shape=(params.map_height, params.map_width), dtype=jnp.int16
    ))
    energy_nodes = jnp.zeros(shape=(params.max_energy_nodes, 2), dtype=jnp.int16)
    energy_nodes_mask = jnp.zeros(shape=(params.max_energy_nodes), dtype=jnp.int16)
    relic_nodes = jnp.zeros(shape=(params.max_relic_nodes, 2), dtype=jnp.int16)
    relic_nodes_mask = jnp.zeros(shape=(params.max_relic_nodes), dtype=jnp.bool)
    if MAP_TYPES[params.map_type] == "dev0":
        # assert params.map_height == 16 and params.map_width == 16
        map_features = set_tile(map_features, 4, 4, NEBULA_TILE)
        map_features = set_tile(map_features, slice(3, 6), slice(2, 4), NEBULA_TILE)
        map_features = set_tile(map_features, slice(4, 7), slice(6, 9), NEBULA_TILE)
        map_features = set_tile(map_features, 4, 5, NEBULA_TILE)
        map_features = set_tile(map_features, slice(9, 12), slice(5, 6), NEBULA_TILE)
        map_features = set_tile(map_features, slice(14, 16), slice(12, 15), NEBULA_TILE)

        map_features = set_tile(map_features, slice(12, 15), slice(8, 10), ASTEROID_TILE)
        map_features = set_tile(map_features, slice(1, 4), slice(6, 8), ASTEROID_TILE)

        map_features = set_tile(map_features, slice(11, 12), slice(3, 6), ASTEROID_TILE)
        map_features = set_tile(map_features, slice(4, 5), slice(10, 13), ASTEROID_TILE)
        map_features = set_tile(map_features,15, 0, ASTEROID_TILE)

        map_features = set_tile(map_features, 11, 11, NEBULA_TILE)
        map_features = set_tile(map_features, 11, 12, NEBULA_TILE)
        energy_nodes = energy_nodes.at[0, :].set(jnp.array([4, 4], dtype=jnp.int16))
        energy_nodes_mask = energy_nodes_mask.at[0].set(1)
        energy_nodes = energy_nodes.at[1, :].set(jnp.array([19, 19], dtype=jnp.int16))
        energy_nodes_mask = energy_nodes_mask.at[1].set(1)
        energy_node_fns = jnp.array(
            [
                [0, 1.2, 1, 4],
                # [1, 4, 0, 2],
                [0, 1.2, 1, 4],
                # [1, 4, 0, 0]
            ]
        )
        energy_node_fns = jnp.concat([energy_node_fns, jnp.zeros((params.max_energy_nodes - 2, 4), dtype=jnp.float32)], axis=0)

    relic_node_configs = (
        jax.random.randint(
            key,
            shape=(
                params.max_relic_nodes,
                params.relic_config_size,
                params.relic_config_size,
            ),
            minval=0,
            maxval=10,
            dtype=jnp.int16,
        )
        >= 6
    )
    # elif params.map_type == "random":
    # Apply the nebula tiles to the map_features
    # map_features = map_features.replace(tile_type=jnp.where(nebula_map, NEBULA_TILE, EMPTY_TILE))
    perlin_noise = generate_perlin_noise_2d( (params.map_height, params.map_width), (4, 4))
    noise = jnp.where(perlin_noise > 0.5, 1, 0)
    noise = noise | noise.T
    # Flip the noise matrix's rows and columns in reverse
    noise = noise[::-1, ::1]
    
    map_features = map_features.replace(tile_type=jnp.where(noise, NEBULA_TILE, 0))
    
    noise = jnp.where(perlin_noise < -0.6, 1, 0)
    noise = noise | noise.T
    # Flip the noise matrix's rows and columns in reverse
    noise = noise[::-1, ::1]
    # jax.debug.breakpoint()
    map_features = map_features.replace(tile_type=jnp.place(map_features.tile_type, noise, 2, inplace=False))
    
    noise = generate_perlin_noise_2d( (params.map_height, params.map_width), (4, 4))
    # Find the positions of the two highest noise values
    flat_indices = jnp.argsort(noise.ravel())[-2:]  # Get indices of two highest values
    highest_positions = jnp.column_stack(jnp.unravel_index(flat_indices, noise.shape))
    
    # Convert to int16 to match the dtype of energy_nodes
    highest_positions = highest_positions.astype(jnp.int16)
    # Set relic nodes to the positions of highest noise values
    relic_nodes = relic_nodes.at[0, :].set(highest_positions[0])
    relic_nodes_mask = relic_nodes_mask.at[0].set(True)
    relic_nodes = relic_nodes.at[1, :].set(highest_positions[1])
    relic_nodes_mask = relic_nodes_mask.at[1].set(True)
    mirrored_pos1 = jnp.array([params.map_width - highest_positions[0][1]-1, params.map_height - highest_positions[0][0]-1], dtype=jnp.int16)
    mirrored_pos2 = jnp.array([params.map_width - highest_positions[1][1]-1, params.map_height - highest_positions[1][0]-1], dtype=jnp.int16)
    # Set the mirrored positions for the other two relic nodes
    relic_nodes = relic_nodes.at[2, :].set(mirrored_pos1)
    relic_nodes_mask = relic_nodes_mask.at[2].set(True)
    relic_nodes = relic_nodes.at[3, :].set(mirrored_pos2)
    relic_nodes_mask = relic_nodes_mask.at[3].set(True)
    relic_node_configs = relic_node_configs.at[2].set(relic_node_configs[0])
    relic_node_configs = relic_node_configs.at[3].set(relic_node_configs[1])
    return dict(
        map_features=map_features,
        energy_nodes=energy_nodes,
        energy_node_fns=energy_node_fns,
        relic_nodes=relic_nodes,
        energy_nodes_mask=energy_nodes_mask,
        relic_nodes_mask=relic_nodes_mask,
        relic_node_configs=relic_node_configs,
    )

import numpy as np
def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)
def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    # print("grid", grid.shape)
    # print("g00", g00.shape)
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
# @functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
# def generate_perlin_noise_2d(
#     key, shape, res, tileable=(False, False), interpolant=interpolant
# ):
#     """Generate a 2D numpy array of perlin noise.

#     Args:
#         shape: The shape of the generated array (tuple of two ints).
#             This must be a multple of res.
#         res: The number of periods of noise to generate along each
#             axis (tuple of two ints). Note shape must be a multiple of
#             res.
#         tileable: If the noise should be tileable along each axis
#             (tuple of two bools). Defaults to (False, False).
#         interpolant: The interpolation function, defaults to
#             t*t*t*(t*(t*6 - 15) + 10).

#     Returns:
#         A numpy array of shape shape with the generated noise.

#     Raises:
#         ValueError: If shape is not a multiple of res.
#     """
#     delta = (res[0] / shape[0], res[1] / shape[1])
#     d = (shape[0] // res[0], shape[1] // res[1])
#     grid = jnp.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
#              .transpose(1, 2, 0) % 1
#     # Gradients
#     # angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
#     angles = 2*jnp.pi*jax.random.uniform(key,minval=res[0]+1, maxval=res[1]+1)
#     gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
#     if tileable[0]:
#         gradients[-1,:] = gradients[0,:]
#     if tileable[1]:
#         gradients[:,-1] = gradients[:,0]
#     gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
#     g00 = gradients[    :-d[0],    :-d[1]]
#     g10 = gradients[d[0]:     ,    :-d[1]]
#     g01 = gradients[    :-d[0],d[1]:     ]
#     g11 = gradients[d[0]:     ,d[1]:     ]
#     # Ramps
#     print(grid.shape)
#     print(g00.shape)
#     n00 = jnp.sum(jnp.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
#     n10 = jnp.sum(jnp.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
#     n01 = jnp.sum(jnp.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
#     n11 = jnp.sum(jnp.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
#     # Interpolation
#     t = interpolant(grid)
#     n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
#     n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
#     return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)