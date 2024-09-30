import functools
from typing import Any, Dict, Optional, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces
from jax import lax

from luxai_s3.params import EnvParams
from luxai_s3.spaces import MultiDiscrete
from luxai_s3.state import ASTEROID_TILE, ENERGY_NODE_FNS, NEBULA_TILE, EnvObs, EnvState, MapTile, UnitState, gen_state, spawn_unit
from luxai_s3.pygame_render import LuxAIPygameRenderer


class LuxAIS3Env(environment.Environment):
    def __init__(self, auto_reset=False, **kwargs):
        super().__init__(**kwargs)
        self.renderer = LuxAIPygameRenderer()
        self.auto_reset = auto_reset

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def compute_energy_features(self, state, params):
        # first compute a array of shape (map_height, map_width, num_energy_nodes) with values equal to the distance of the tile to the energy node
        mm = jnp.meshgrid(jnp.arange(params.map_width), jnp.arange(params.map_height))
        mm = jnp.stack([mm[0], mm[1]]).T # mm[x, y] gives [x, y]
        distances_to_nodes = jax.vmap(lambda pos: jnp.linalg.norm(mm - pos, axis=-1))(state.energy_nodes)
        def compute_energy_field(node_fn_spec, distances_to_node, mask):
            fn_i, x, y, z = node_fn_spec
            return jnp.where(mask, lax.switch(fn_i.astype(jnp.int16), ENERGY_NODE_FNS, distances_to_node, x, y, z), jnp.zeros_like(distances_to_node))
        energy_field = jax.vmap(compute_energy_field)(state.energy_node_fns, distances_to_nodes, state.energy_nodes_mask)
        energy_field = jnp.where(energy_field.mean() < .25, energy_field + (.25 - energy_field.mean()), energy_field)
        energy_field = jnp.round(energy_field.sum(0))
        energy_field = jnp.clip(energy_field, params.min_energy_per_tile, params.max_energy_per_tile)
        state = state.replace(map_features=state.map_features.replace(energy=energy_field))
        return state
    
    def compute_sensor_masks(self, state, params):
        """Compute the vision power and sensor mask for both teams 
        
        Algorithm:

        For each team, generate a integer vision power array over the map. 
        For each unit in team, add unit sensor range value (its kind of like the units sensing power/depth) to each tile the unit's sensor range
        Clamp the vision power array to range [0, unit_sensing_range].

        With 2 vision power maps, take the nebula vision mask * nebula vision power and subtract it from the vision power maps.
        Now any time the vision power map has value > 0, the team can sense the tile. This forms the sensor mask
        """

        vision_power_map_padding = params.unit_sensor_range
        vision_power_map = jnp.zeros(
            shape=(params.num_teams, params.map_height + 2 * vision_power_map_padding, params.map_width + 2 * vision_power_map_padding),
            dtype=jnp.int16,
        )

        # Update sensor mask based on the sensor range
        def update_vision_power_map(unit_pos, sensor_mask):
            x, y = unit_pos
            update = jnp.ones(shape= (params.unit_sensor_range * 2 + 1, params.unit_sensor_range * 2 + 1), dtype=jnp.int16)
            for i in range(params.unit_sensor_range + 1):
                update = update.at[i:params.unit_sensor_range * 2 + 1 - i, i:params.unit_sensor_range * 2 + 1 - i].set(i + 1)
            x, y = unit_pos
            sensor_mask = jax.lax.dynamic_update_slice(
                sensor_mask,
                update=update,
                start_indices=(
                    x - params.unit_sensor_range + vision_power_map_padding,
                    y - params.unit_sensor_range + vision_power_map_padding,
                ),
            )
            return sensor_mask

        # Apply the sensor mask update for all units of both teams
        def update_unit_vision_power_map(unit_pos, mask, sensor_mask):
            return jax.lax.cond(
                mask,
                lambda: update_vision_power_map(unit_pos, sensor_mask),
                lambda: sensor_mask,
            )

        def update_team_vision_power_map(team_units, team_mask, sensor_mask):
            def body_fun(carry, i):
                sensor_mask = carry
                return (
                    update_unit_vision_power_map(
                        team_units.position[i], team_mask[i], sensor_mask
                    ),
                    None,
                )

            final_sensor_mask, _ = jax.lax.scan(
                body_fun, sensor_mask, jnp.arange(params.max_units)
            )
            return final_sensor_mask

        vision_power_map = jax.vmap(update_team_vision_power_map)(
            state.units, state.units_mask, vision_power_map
        )
        vision_power_map = vision_power_map[:, vision_power_map_padding:-vision_power_map_padding, vision_power_map_padding:-vision_power_map_padding]

        # handle nebula tiles
        vision_power_map = vision_power_map - (state.map_features.tile_type == NEBULA_TILE) * params.nebula_tile_vision_reduction
        
        sensor_mask = vision_power_map > 0
        state = state.replace(sensor_mask=sensor_mask)
        state = state.replace(vision_power_map=vision_power_map)
        return state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        
        state = self.compute_energy_features(state, params)

        # state = state.replace() # TODO (stao)
        action = jnp.stack([action["player_0"], action["player_1"]])
        ### process unit movement ###
        # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left
        # Define movement directions
        directions = jnp.array(
            [
                [0, 0],  # Do nothing
                [0, -1],  # Move up
                [1, 0],  # Move right
                [0, 1],  # Move down
                [-1, 0],  # Move left
            ],
            dtype=jnp.int16,
        )

        def move_unit(unit: UnitState, action, mask):
            new_pos = unit.position + directions[action]
            # Check if the new position is on a map feature of value 2
            is_blocked = state.map_features.tile_type[new_pos[0], new_pos[1]] == ASTEROID_TILE
            enough_energy = unit.energy >= params.unit_move_cost
            # If blocked, keep the original position
            # new_pos = jnp.where(is_blocked, unit.position, new_pos)
            # Ensure the new position is within the map boundaries
            new_pos = jnp.clip(
                new_pos,
                0,
                jnp.array(
                    [params.map_width - 1, params.map_height - 1], dtype=jnp.int16
                ),
            )
            unit_moved = mask & ~is_blocked & enough_energy
            # Update the unit's position only if it's active
            # import ipdb; ipdb.set_trace()
            return UnitState(position=jnp.where(unit_moved, new_pos, unit.position), energy=jnp.where(unit_moved, unit.energy - params.unit_move_cost, unit.energy))

        # Move units for both teams
        state = state.replace(
            units=jax.vmap(
                lambda team_units, team_action, team_mask: jax.vmap(move_unit, in_axes=(0, 0, 0))(
                    team_units, team_action, team_mask
                ), in_axes=(0, 0, 0)
            )(state.units, action, state.units_mask)
        )

        """apply energy field to the units"""
        # Update unit energy based on the energy field of their current position
        def update_unit_energy(unit: UnitState, mask):
            x, y = unit.position
            energy_gain = state.map_features.energy[x, y] - (state.map_features.tile_type[x, y] == NEBULA_TILE) * params.nebula_tile_energy_reduction
            new_energy = jnp.clip(unit.energy + energy_gain, params.min_unit_energy, params.max_unit_energy)
            return UnitState(position=unit.position, energy=jnp.where(mask, new_energy, unit.energy))

        # Apply the energy update for all units of both teams
        state = state.replace(
            units=jax.vmap(
                lambda team_units, team_mask: jax.vmap(update_unit_energy)(
                    team_units, team_mask
                )
            )(state.units, state.units_mask)
        )

        state = self.compute_sensor_masks(state, params)
        
        # Shift objects around in space
        # Move the nebula tiles in state.map_features.tile_types up by 1 and to the right by 1
        # this is also symmetric nebula tile movement
        new_tile_types_map = jnp.roll(state.map_features.tile_type, shift=(1 * jnp.sign(params.nebula_tile_drift_speed), -1 * jnp.sign(params.nebula_tile_drift_speed)), axis=(0, 1))
        new_tile_types_map = jnp.where(state.steps * params.nebula_tile_drift_speed % 1 == 0, new_tile_types_map, state.map_features.tile_type)
        # new_energy_nodes = state.energy_nodes + jnp.array([1 * jnp.sign(params.energy_node_drift_speed), -1 * jnp.sign(params.energy_node_drift_speed)])
        
        energy_node_deltas = jnp.round(jax.random.uniform(key=key, shape=(params.max_energy_nodes, 2), minval=-params.energy_node_drift_magnitude, maxval=params.energy_node_drift_magnitude)).astype(jnp.int16)
        # TODO symmetric movement
        # energy_node_deltas = jnp.round(jax.random.uniform(key=key, shape=(params.max_energy_nodes // 2, 2), minval=-params.energy_node_drift_magnitude, maxval=params.energy_node_drift_magnitude)).astype(jnp.int16)
        # energy_node_deltas = jnp.concatenate((energy_node_deltas, energy_node_deltas[::-1]))
        new_energy_nodes = jnp.clip(state.energy_nodes + energy_node_deltas, min=jnp.array([0, 0]), max=jnp.array([params.map_width, params.map_height]))
        new_energy_nodes = jnp.where(state.steps * params.energy_node_drift_speed % 1 == 0, new_energy_nodes, state.energy_nodes)
        state = state.replace(map_features=state.map_features.replace(tile_type=new_tile_types_map), energy_nodes=new_energy_nodes)

        
        # Compute relic scores
        def compute_relic_score(unit, relic_nodes_map_weights, mask):
            total_score = relic_nodes_map_weights[unit.position[0], unit.position[1]]
            return total_score & mask

        def team_relic_score(units, units_mask):
            scores = jax.vmap(compute_relic_score, in_axes=(0, None, 0))(
                units,
                state.relic_nodes_map_weights,
                units_mask,
            )
            return jnp.sum(scores, dtype=jnp.int32)

        team_scores = jax.vmap(team_relic_score)(state.units, state.units_mask)
        # team_1_score = team_relic_score(state.units[1], state.units_mask[1])
        # jax.debug.print("{team_scores}", team_scores=team_scores)
        # Update team points
        state = state.replace(
            team_points=state.team_points + team_scores
        )
        reward = dict()
        for k in range(params.num_teams):
            reward[f"player_{k}"] = team_scores[k]
        terminated = self.is_terminal(state, params)

        # if match ended, then remove all units
        match_ended = state.match_steps >= params.max_steps_in_match
        state = state.replace(units_mask=jnp.where(match_ended, jnp.zeros_like(state.units_mask), state.units_mask), match_steps=jnp.where(match_ended, 0, state.match_steps))

        # spawn units in
        spawn_units_in = (state.match_steps % params.spawn_rate == 0)
        # TODO (stao): only logic in code that probably doesn't not handle more than 2 teams, everything else is vmapped across teams
        def spawn_team_units(state: EnvState):
            state = spawn_unit(state, 0, state.units_mask[0].sum(), [0, 0], params)
            state = spawn_unit(state, 1, state.units_mask[1].sum(), [params.map_width - 1, params.map_height - 1], params)
            return state
        state = jax.lax.cond(spawn_units_in, lambda: spawn_team_units(state), lambda: state)

        # Update state's step count
        state = state.replace(steps=state.steps + 1, match_steps=state.match_steps + 1)
        truncated = state.steps >= params.max_steps_in_match * params.match_count_per_episode
        return (
            lax.stop_gradient(self.get_obs(state, params, key=key)),
            lax.stop_gradient(state),
            reward,
            terminated,
            truncated,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[EnvObs, EnvState]:
        """Reset environment state by sampling initial position."""

        state = gen_state(key=key, params=params)
        state = self.compute_energy_features(state, params)
        state = self.compute_sensor_masks(state, params)

        return self.get_obs(state, params=params, key=key), state

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        print("Compiled step")
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, truncated, info = self.step_env(
            key, state, action, params
        )
        info["final_state"] = state_st
        info["final_observation"] = obs_st
        if self.auto_reset:
            done = terminated | truncated
            obs_re, state_re = self.reset_env(key_reset, params)
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)
        else:
            obs = obs_st
            state = state_st
        # Auto-reset environment based on done
        done = terminated | truncated
        
        # all agents terminate/truncate at same time
        terminated_dict = dict()
        truncated_dict = dict()
        for k in range(params.num_teams):
            terminated_dict[f"player_{k}"] = terminated
            truncated_dict[f"player_{k}"] = truncated
            info[f"player_{k}"] = dict()
        return obs, state, reward, terminated_dict, truncated_dict, info

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        print("Compiled reset")
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    # @functools.partial(jax.jit, static_argnums=(0, 2))
    def get_obs(self, state: EnvState, params=None, key=None) -> EnvObs:
        """Return observation from raw state, handling partial observability."""
        # determine correct unit mask for units that are visible
        # unit_mask = jnp.logical_and(state.units_mask, jax.vmap(lambda t: state.sensor_mask[t, state.units.position[t, u, 0], state.units.position[t, u, 1]])(jnp.arange(state.units.position.shape[0]))) for u in range(state.units.position.shape[1]))
        
        obs = dict()
        
        def update_unit_mask(unit_position, unit_mask, sensor_mask):
            return unit_mask & sensor_mask[unit_position[0], unit_position[1]]
        def update_team_unit_mask(unit_position, unit_mask, sensor_mask):
            return jax.vmap(update_unit_mask, in_axes=(0, 0, None))(unit_position, unit_mask, sensor_mask)
       
            
        for t in range(params.num_teams):
            other_team_ids = jnp.array([t2 for t2 in range(params.num_teams) if t2 != t])
            new_unit_masks = jax.vmap(update_team_unit_mask, in_axes=(0, 0, None))(state.units.position[other_team_ids], state.units_mask[other_team_ids], state.sensor_mask[t])
            new_unit_masks = state.units_mask.at[other_team_ids].set(new_unit_masks)
            team_obs = EnvObs(
                units=UnitState(
                    position=jnp.where(new_unit_masks[..., None], state.units.position, -1),
                    energy=jnp.where(new_unit_masks[..., None], state.units.energy, -1),
                ),
                units_mask=new_unit_masks,
                sensor_mask=state.sensor_mask[t],
                map_features=MapTile(
                    energy=jnp.where(state.sensor_mask[t], state.map_features.energy, -1),
                    tile_type=jnp.where(state.sensor_mask[t], state.map_features.tile_type, -1),
                ),
                team_points=state.team_points,
                steps=state.steps,
                match_steps=state.match_steps
            )
            obs[f"player_{t}"] = team_obs
        return obs

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal. This occurs when either team wins/loses outright."""
        terminated = jnp.array(False)
        return terminated

    @property
    def name(self) -> str:
        """Environment name."""
        return "Lux AI Season 3"

    def render(self, state: EnvState, params: EnvParams):
        self.renderer.render(state, params)

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        size = np.ones(params.max_units) * 5
        return spaces.Dict(dict(player_0=MultiDiscrete(size), player_1=MultiDiscrete(size)))

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(10)

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Discrete(10)
