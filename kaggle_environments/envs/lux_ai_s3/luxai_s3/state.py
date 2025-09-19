import functools

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from luxai_s3.params import MAP_TYPES, EnvParams

EMPTY_TILE = 0
NEBULA_TILE = 1
ASTEROID_TILE = 2

ENERGY_NODE_FNS = [lambda d, x, y, z: jnp.sin(d * x + y) * z, lambda d, x, y, z: (x / (d + 1) + y) * z]


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
    """Mask of relic nodes in the environment with shape (N, ) for N max relic nodes"""
    relic_nodes_map_weights: chex.Array
    """Map of relic nodes in the environment with shape (H, W) for H height, W width. Each element is equal to the 1-indexed id of the relic node. This is generated from other state"""

    relic_spawn_schedule: chex.Array
    """Relic spawn schedule in the environment with shape (N, ) for N max relic nodes. Elements are the game timestep at which the relic node spawns"""

    map_features: MapTile
    """Map features in the environment with shape (W, H, 2) for W width, H height
    """

    sensor_mask: chex.Array
    """Sensor mask in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    vision_power_map: chex.Array
    """Vision power map in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""
    team_wins: chex.Array
    """Team wins in the environment with shape (T) for T teams"""

    steps: int = 0
    """steps taken in the environment"""
    match_steps: int = 0
    """steps taken in the current match"""


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
    relic_nodes: chex.Array
    """Position of all relic nodes with shape (N, 2) for N max relic nodes and 2 features for position (x, y). Number is -1 if not visible"""
    relic_nodes_mask: chex.Array
    """Mask of all relic nodes with shape (N) for N max relic nodes"""
    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""
    team_wins: chex.Array
    """Team wins in the environment with shape (T) for T teams"""
    steps: int = 0
    """steps taken in the environment"""
    match_steps: int = 0
    """steps taken in the current match"""


def serialize_env_states(env_states: list[EnvState]):
    def serialize_array(root: EnvState, arr, key_path: str = ""):
        if key_path in [
            "sensor_mask",
            "relic_nodes_mask",
            "energy_nodes_mask",
            "energy_node_fns",
            "relic_nodes_map_weights",
            "relic_spawn_schedule",
        ]:
            return None
        if key_path == "relic_nodes":
            return root.relic_nodes[root.relic_nodes_mask].tolist()
        if key_path == "relic_node_configs":
            return root.relic_node_configs[root.relic_nodes_mask].tolist()
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


def state_to_flat_obs(state: EnvState) -> chex.Array:
    pass


def flat_obs_to_state(flat_obs: chex.Array) -> EnvState:
    pass


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def gen_state(
    key: chex.PRNGKey,
    env_params: EnvParams,
    max_units: int,
    num_teams: int,
    map_type: int,
    map_width: int,
    map_height: int,
    max_energy_nodes: int,
    max_relic_nodes: int,
    relic_config_size: int,
) -> EnvState:
    generated = gen_map(
        key, env_params, map_type, map_width, map_height, max_energy_nodes, max_relic_nodes, relic_config_size
    )
    relic_nodes_map_weights = jnp.zeros(shape=(map_width, map_height), dtype=jnp.int16)

    # TODO (this could be optimized better)
    def update_relic_node(relic_nodes_map_weights, relic_data):
        relic_node, relic_node_config, mask, relic_node_id = relic_data
        start_y = relic_node[1] - relic_config_size // 2
        start_x = relic_node[0] - relic_config_size // 2

        for dy in range(relic_config_size):
            for dx in range(relic_config_size):
                y, x = start_y + dy, start_x + dx
                valid_pos = jnp.logical_and(
                    jnp.logical_and(y >= 0, x >= 0),
                    jnp.logical_and(y < map_height, x < map_width),
                )
                # ensure we don't override previous spawns
                has_points = jnp.logical_and(relic_nodes_map_weights > 0, relic_nodes_map_weights <= relic_node_id + 1)
                relic_nodes_map_weights = jnp.where(
                    valid_pos & mask & jnp.logical_not(has_points) & relic_node_config[dx, dy],
                    relic_nodes_map_weights.at[x, y].set(
                        relic_node_config[dx, dy].astype(jnp.int16) * (relic_node_id + 1)
                    ),
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
            jnp.arange(max_relic_nodes, dtype=jnp.int16) % (max_relic_nodes // 2),
        ),
    )

    state = EnvState(
        units=UnitState(
            position=jnp.zeros(shape=(num_teams, max_units, 2), dtype=jnp.int16),
            energy=jnp.zeros(shape=(num_teams, max_units, 1), dtype=jnp.int16),
        ),
        units_mask=jnp.zeros(shape=(num_teams, max_units), dtype=jnp.bool),
        team_points=jnp.zeros(shape=(num_teams), dtype=jnp.int32),
        team_wins=jnp.zeros(shape=(num_teams), dtype=jnp.int32),
        energy_nodes=generated["energy_nodes"],
        energy_node_fns=generated["energy_node_fns"],
        energy_nodes_mask=generated["energy_nodes_mask"],
        # energy_field=jnp.zeros(shape=(params.map_height, params.map_width), dtype=jnp.int16),
        relic_nodes=generated["relic_nodes"],
        relic_nodes_mask=jnp.zeros(
            shape=(max_relic_nodes), dtype=jnp.bool
        ),  # as relic nodes are spawn in, we start with them all invisible.
        relic_node_configs=generated["relic_node_configs"],
        relic_nodes_map_weights=relic_nodes_map_weights,
        relic_spawn_schedule=generated["relic_spawn_schedule"],
        sensor_mask=jnp.zeros(
            shape=(num_teams, map_height, map_width),
            dtype=jnp.bool,
        ),
        vision_power_map=jnp.zeros(shape=(num_teams, map_height, map_width), dtype=jnp.int16),
        map_features=generated["map_features"],
    )
    return state


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def gen_map(
    key: chex.PRNGKey,
    params: EnvParams,
    map_type: int,
    map_height: int,
    map_width: int,
    max_energy_nodes: int,
    max_relic_nodes: int,
    relic_config_size: int,
) -> chex.Array:
    map_features = MapTile(
        energy=jnp.zeros(shape=(map_height, map_width), dtype=jnp.int16),
        tile_type=jnp.zeros(shape=(map_height, map_width), dtype=jnp.int16),
    )
    energy_nodes = jnp.zeros(shape=(max_energy_nodes, 2), dtype=jnp.int16)
    energy_nodes_mask = jnp.zeros(shape=(max_energy_nodes), dtype=jnp.bool)
    relic_nodes = jnp.zeros(shape=(max_relic_nodes, 2), dtype=jnp.int16)
    relic_nodes_mask = jnp.zeros(shape=(max_relic_nodes), dtype=jnp.bool)

    if MAP_TYPES[map_type] == "random":
        ### Generate nebula tiles ###
        key, subkey = jax.random.split(key)
        perlin_noise = generate_perlin_noise_2d(subkey, (map_height, map_width), (4, 4))
        noise = jnp.where(perlin_noise > 0.5, 1, 0)
        # mirror along diagonal
        noise = noise | noise.T
        noise = noise[::-1, ::1]
        map_features = map_features.replace(tile_type=jnp.where(noise, NEBULA_TILE, 0))

        ### Generate asteroid tiles ###
        key, subkey = jax.random.split(key)
        perlin_noise = generate_perlin_noise_2d(subkey, (map_height, map_width), (8, 8))
        noise = jnp.where(perlin_noise < -0.5, 1, 0)
        # mirror along diagonal
        noise = noise | noise.T
        noise = noise[::-1, ::1]
        map_features = map_features.replace(
            tile_type=jnp.place(map_features.tile_type, noise, ASTEROID_TILE, inplace=False)
        )

        ### Generate relic nodes ###
        key, subkey = jax.random.split(key)
        noise = generate_perlin_noise_2d(subkey, (map_height, map_width), (4, 4))
        # Find the positions of the  highest noise values
        flat_indices = jnp.argsort(noise.ravel())[-max_relic_nodes // 2 :]  # Get indices of two highest values
        highest_positions = jnp.column_stack(jnp.unravel_index(flat_indices, noise.shape))

        # relic nodes have a fixed density of 20% nearby tiles can yield points
        relic_node_configs = (
            jax.random.randint(
                key,
                shape=(
                    max_relic_nodes,
                    relic_config_size,
                    relic_config_size,
                ),
                minval=0,
                maxval=10,
            ).astype(jnp.float32)
            >= 7.5
        )
        highest_positions = highest_positions.astype(jnp.int16)
        mirrored_positions = jnp.stack(
            [map_width - highest_positions[:, 1] - 1, map_height - highest_positions[:, 0] - 1],
            dtype=jnp.int16,
            axis=-1,
        )
        relic_nodes = jnp.concat([highest_positions, mirrored_positions], axis=0)

        key, subkey = jax.random.split(key)
        num_spawned_relic_nodes = jax.random.randint(key, (1,), minval=1, maxval=(max_relic_nodes // 2) + 1)
        relic_nodes_mask_half = jnp.arange(max_relic_nodes // 2) < num_spawned_relic_nodes
        relic_nodes_mask = jnp.concat([relic_nodes_mask_half, relic_nodes_mask_half], axis=0)
        relic_node_configs = relic_node_configs.at[max_relic_nodes // 2 :].set(
            relic_node_configs[: max_relic_nodes // 2].transpose(0, 2, 1)[:, ::-1, ::-1]
        )
        # note that relic nodes mask is always increasing.

        ### Generate energy nodes ###
        key, subkey = jax.random.split(key)
        noise = generate_perlin_noise_2d(subkey, (map_height, map_width), (4, 4))
        # Find the positions of the  highest noise values
        flat_indices = jnp.argsort(noise.ravel())[-max_energy_nodes // 2 :]  # Get indices of highest values
        highest_positions = jnp.column_stack(jnp.unravel_index(flat_indices, noise.shape)).astype(jnp.int16)
        mirrored_positions = jnp.stack(
            [map_width - highest_positions[:, 1] - 1, map_height - highest_positions[:, 0] - 1],
            dtype=jnp.int16,
            axis=-1,
        )
        energy_nodes = jnp.concat([highest_positions, mirrored_positions], axis=0)
        key, subkey = jax.random.split(key)
        energy_nodes_mask_half = jax.random.randint(key, (max_energy_nodes // 2,), minval=0, maxval=2).astype(jnp.bool)
        energy_nodes_mask_half = energy_nodes_mask_half.at[0].set(True)
        energy_nodes_mask = energy_nodes_mask.at[: max_energy_nodes // 2].set(energy_nodes_mask_half)
        energy_nodes_mask = energy_nodes_mask.at[max_energy_nodes // 2 :].set(energy_nodes_mask_half)

        energy_node_fns = jnp.array(
            [
                [0, 1.2, 1, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                # [1, 4, 0, 2],
                [0, 1.2, 1, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                # [1, 4, 0, 0]
            ]
        )

        # generate a random relic spawn schedule
        # if number is -1, then relic node is never spawned, otherwise spawn at that game timestep
        assert max_relic_nodes == 6, "random map generation is hardcoded to use 6 relic nodes at most per map"
        key, subkey = jax.random.split(key)
        relic_spawn_schedule_half = jax.random.randint(
            key, (max_relic_nodes // 2,), minval=0, maxval=params.max_steps_in_match // 2
        ) + jnp.arange(3) * (params.max_steps_in_match + 1)
        relic_spawn_schedule = jnp.concat([relic_spawn_schedule_half, relic_spawn_schedule_half], axis=0)
        relic_spawn_schedule = jnp.where(relic_nodes_mask, relic_spawn_schedule, -1)

    return dict(
        map_features=map_features,
        energy_nodes=energy_nodes,
        energy_node_fns=energy_node_fns,
        relic_nodes=relic_nodes,
        energy_nodes_mask=energy_nodes_mask,
        relic_nodes_mask=relic_nodes_mask,
        relic_node_configs=relic_node_configs,
        relic_spawn_schedule=relic_spawn_schedule,
    )


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_perlin_noise_2d(key, shape, res, tileable=(False, False), interpolant=interpolant):
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
    grid = jnp.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * jnp.pi * jax.random.uniform(key, (res[0] + 1, res[1] + 1))
    gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    # Ramps
    n00 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return jnp.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
