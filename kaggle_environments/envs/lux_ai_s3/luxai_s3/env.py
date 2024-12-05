import functools
from typing import Any, Dict, Optional, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces
from jax import lax

from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.spaces import MultiDiscrete
from luxai_s3.state import (
    ASTEROID_TILE,
    ENERGY_NODE_FNS,
    NEBULA_TILE,
    EnvObs,
    EnvState,
    MapTile,
    UnitState,
    gen_state
)
from luxai_s3.pygame_render import LuxAIPygameRenderer


class LuxAIS3Env(environment.Environment):
    def __init__(
        self, auto_reset=False, fixed_env_params: EnvParams = EnvParams(), **kwargs
    ):
        super().__init__(**kwargs)
        self.renderer = LuxAIPygameRenderer()
        self.auto_reset = auto_reset
        self.fixed_env_params = fixed_env_params
        """fixed env params for concrete/static values. Necessary for jit/vmap capability with randomly sampled maps which must of consistent shape"""

    @property
    def default_params(self) -> EnvParams:
        params = EnvParams()
        params = jax.tree_map(jax.numpy.array, params)
        return params

    def compute_unit_counts_map(self, state: EnvState, params: EnvParams, exclude_negative_energy_units: bool = False):
        # map of total units per team on each tile, shape (num_teams, map_width, map_height)
        unit_counts_map = jnp.zeros(
            (self.fixed_env_params.num_teams, self.fixed_env_params.map_width, self.fixed_env_params.map_height), dtype=jnp.int16
        )

        def update_unit_counts_map(unit_position, unit_mask, unit_energy_nonnegative, unit_counts_map):
            if exclude_negative_energy_units:
                mask = unit_mask & unit_energy_nonnegative
            else:
                mask = unit_mask
            unit_counts_map = unit_counts_map.at[
                unit_position[0], unit_position[1]
            ].add(mask.astype(jnp.int16))
            return unit_counts_map

        for t in range(self.fixed_env_params.num_teams):
            unit_counts_map = unit_counts_map.at[t].add(
                jnp.sum(
                    jax.vmap(update_unit_counts_map, in_axes=(0, 0, 0, None), out_axes=0)(
                        state.units.position[t], state.units_mask[t], state.units.energy[t, :, 0] >= 0, unit_counts_map[t]
                    ),
                    axis=0,
                    dtype=jnp.int16
                )
            )
        return unit_counts_map

    def compute_energy_features(self, state: EnvState, params: EnvParams):
        # first compute a array of shape (map_height, map_width, num_energy_nodes) with values equal to the distance of the tile to the energy node
        mm = jnp.meshgrid(jnp.arange(self.fixed_env_params.map_width), jnp.arange(self.fixed_env_params.map_height))
        mm = jnp.stack([mm[0], mm[1]]).T.astype(jnp.int16)  # mm[x, y] gives [x, y]
        distances_to_nodes = jax.vmap(lambda pos: jnp.linalg.norm(mm - pos, axis=-1))(
            state.energy_nodes
        )

        def compute_energy_field(node_fn_spec, distances_to_node, mask):
            fn_i, x, y, z = node_fn_spec
            return jnp.where(
                mask,
                lax.switch(
                    fn_i.astype(jnp.int16), ENERGY_NODE_FNS, distances_to_node, x, y, z
                ),
                jnp.zeros_like(distances_to_node),
            )

        energy_field = jax.vmap(compute_energy_field)(
            state.energy_node_fns, distances_to_nodes, state.energy_nodes_mask
        )
        energy_field = jnp.where(
            energy_field.mean() < 0.25,
            energy_field + (0.25 - energy_field.mean()),
            energy_field,
        )
        energy_field = jnp.round(energy_field.sum(0)).astype(jnp.int16)
        energy_field = jnp.clip(
            energy_field, params.min_energy_per_tile, params.max_energy_per_tile
        )
        state = state.replace(
            map_features=state.map_features.replace(energy=energy_field)
        )
        return state

    def compute_sensor_masks(self, state, params: EnvParams):
        """Compute the vision power and sensor mask for both teams

        Algorithm:

        For each team, generate a integer vision power array over the map.
        For each unit in team, add unit sensor range value (its kind of like the units sensing power/depth) to each tile the unit's sensor range
        Clamp the vision power array to range [0, unit_sensing_range].

        With 2 vision power maps, take the nebula vision mask * nebula vision power and subtract it from the vision power maps.
        Now any time the vision power map has value > 0, the team can sense the tile. This forms the sensor mask
        """

        max_sensor_range = env_params_ranges["unit_sensor_range"][-1]
        vision_power_map_padding = max_sensor_range
        vision_power_map = jnp.zeros(
            shape=(
                self.fixed_env_params.num_teams,
                self.fixed_env_params.map_height + 2 * vision_power_map_padding,
                self.fixed_env_params.map_width + 2 * vision_power_map_padding,
            ),
            dtype=jnp.int16,
        )

        # Update sensor mask based on the sensor range
        def update_vision_power_map(unit_pos, vision_power_map):
            x, y = unit_pos
            existing_vision_power = jax.lax.dynamic_slice(
                vision_power_map,
                start_indices=(
                    x - max_sensor_range + vision_power_map_padding,
                    y - max_sensor_range + vision_power_map_padding,
                ),
                slice_sizes=(
                    max_sensor_range * 2 + 1,
                    max_sensor_range * 2 + 1,
                ),
            )
            update = jnp.zeros_like(existing_vision_power)
            for i in range(max_sensor_range + 1):
                val = jnp.where(i > max_sensor_range - params.unit_sensor_range - 1, i + 1 - (max_sensor_range - params.unit_sensor_range), 0).astype(jnp.int16)
                update = update.at[
                    i : max_sensor_range * 2 + 1 - i,
                    i : max_sensor_range * 2 + 1 - i,
                ].set(val)
            vision_power_map = jax.lax.dynamic_update_slice(
                vision_power_map,
                update=update + existing_vision_power,
                start_indices=(
                    x - max_sensor_range + vision_power_map_padding,
                    y - max_sensor_range + vision_power_map_padding,
                ),
            )
            return vision_power_map

        # Apply the sensor mask update for all units of both teams
        def update_unit_vision_power_map(unit_pos, unit_mask, vision_power_map):
            return jax.lax.cond(
                unit_mask,
                lambda: update_vision_power_map(unit_pos, vision_power_map),
                lambda: vision_power_map,
            )

        def update_team_vision_power_map(team_units, unit_mask, vision_power_map):
            def body_fun(carry, i):
                vision_power_map = carry
                return (
                    update_unit_vision_power_map(
                        team_units.position[i], unit_mask[i], vision_power_map
                    ),
                    None,
                )

            vision_power_map, _ = jax.lax.scan(
                body_fun, vision_power_map, jnp.arange(self.fixed_env_params.max_units)
            )
            return vision_power_map

        vision_power_map = jax.vmap(update_team_vision_power_map)(
            state.units, state.units_mask, vision_power_map
        )
        vision_power_map = vision_power_map[
            :,
            vision_power_map_padding:-vision_power_map_padding,
            vision_power_map_padding:-vision_power_map_padding,
        ]
        # handle nebula tiles
        vision_power_map = (
            vision_power_map
            - (state.map_features.tile_type == NEBULA_TILE).astype(jnp.int16)
            * params.nebula_tile_vision_reduction
        )

        sensor_mask = vision_power_map > 0
        state = state.replace(sensor_mask=sensor_mask)
        state = state.replace(vision_power_map=vision_power_map)
        return state

    # @functools.partial(jax.jit, static_argnums=(0, 4))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:

        state = self.compute_energy_features(state, params)

        action = jnp.stack([action["player_0"], action["player_1"]])

        # remove all units if the match ended in the previous step indicated by a reset of match_steps to 0
        state = state.replace(
            units_mask=jnp.where(
                state.match_steps == 0,
                jnp.zeros_like(state.units_mask),
                state.units_mask,
            )
        )
        """remove units that have less than 0 energy"""
        # we remove units at the start of the timestep so that the visualizer can show the unit with negative energy and is marked for removal soon.
        state = state.replace(
            units_mask=(state.units.energy[..., 0] >= 0) & state.units_mask
        )

        """ process unit movement """
        # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
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
            is_blocked = (
                state.map_features.tile_type[new_pos[0], new_pos[1]] == ASTEROID_TILE
            )
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
            unit_moved = (
                mask & ~is_blocked & enough_energy & (action < 5) & (action > 0)
            )
            # Update the unit's position only if it's active. Note energy is used if unit tries to move off map. Energy is not used if unit tries to move into an asteroid tile.
            return UnitState(
                position=jnp.where(unit_moved, new_pos, unit.position),
                energy=jnp.where(
                    unit_moved, unit.energy - params.unit_move_cost, unit.energy
                ),
            )

        # Move units for both teams
        move_actions = action[..., 0]
        state = state.replace(
            units=jax.vmap(
                lambda team_units, team_action, team_mask: jax.vmap(
                    move_unit, in_axes=(0, 0, 0)
                )(team_units, team_action, team_mask),
                in_axes=(0, 0, 0),
            )(state.units, move_actions, state.units_mask)
        )

        original_unit_energy = state.units.energy
        """original amount of energy of all units"""

        """apply sap actions"""
        sap_action_mask = action[..., 0] == 5
        sap_action_deltas = action[..., 1:]

        def sap_unit(
            current_energy: jnp.ndarray,
            all_units: UnitState,
            sap_action_mask,
            sap_action_deltas,
            units_mask,
        ):
            # TODO (stao): clean up this code. It is probably slower than it needs be and could be vmapped perhaps.
            for t in range(self.fixed_env_params.num_teams):
                other_team_ids = jnp.array(
                    [t2 for t2 in range(self.fixed_env_params.num_teams) if t2 != t]
                )
                team_sap_action_deltas = sap_action_deltas[t]  # (max_units, 2)
                team_sap_action_mask = sap_action_mask[t]
                other_team_unit_mask = units_mask[
                    other_team_ids
                ]  # (other_teams, max_units)
                team_sapped_positions = (
                    all_units.position[t] + team_sap_action_deltas
                )  # (max_units, 2)
                # whether the unit is really sapping or not (needs to exist, have enough energy, and a valid sap action)
                team_unit_sapped = (
                    units_mask[t]
                    & team_sap_action_mask
                    & (current_energy[t, :, 0] >= params.unit_sap_cost)
                    & (
                        jnp.max(jnp.abs(team_sap_action_deltas), axis=-1)
                        <= params.unit_sap_range
                    )
                )  # (max_units)
                team_unit_sapped = (
                    team_unit_sapped
                    & (team_sapped_positions >= 0).all(-1)
                    & (team_sapped_positions[:, 0] < self.fixed_env_params.map_width)
                    & (team_sapped_positions[:, 1] < self.fixed_env_params.map_height)
                )
                # the number of times other units are sapped
                other_units_sapped_count = jnp.sum(
                    team_unit_sapped[None, None, :]
                    & jnp.all(
                        all_units.position[other_team_ids][:, :, None]
                        == team_sapped_positions[None],
                        axis=-1,
                    ),
                    axis=-1,
                    dtype=jnp.int16,
                )  # (len(other_team_ids), max_units)
                # remove unit_sap_cost energy from opposition units that were in the middle of a sap action.
                all_units = all_units.replace(
                    energy=all_units.energy.at[other_team_ids].set(
                        jnp.where(
                            other_team_unit_mask[:, :, None]
                            & (other_units_sapped_count[:, :, None] > 0),
                            all_units.energy[other_team_ids]
                            - params.unit_sap_cost
                            * other_units_sapped_count[:, :, None],
                            all_units.energy[other_team_ids],
                        )
                    )
                )

                # remove unit_sap_cost * unit_sap_dropoff_factor energy from opposition units that were on tiles adjacent to the center of a sap action.
                adjacent_offsets = jnp.array(
                    [
                        [-1, -1],
                        [-1, 0],
                        [-1, 1],
                        [0, -1],
                        [0, 1],
                        [1, -1],
                        [1, 0],
                        [1, 1],
                    ], dtype=jnp.int16
                )
                team_sapped_adjacent_positions = (
                    team_sapped_positions[:, None, :] + adjacent_offsets
                )  # (max_units, len(adjacent_offsets), 2)
                other_units_adjacent_sapped_count = jnp.sum(
                    team_unit_sapped[None, None, :, None]
                    & jnp.all(
                        all_units.position[other_team_ids][:, :, None, None]
                        == team_sapped_adjacent_positions[None],
                        axis=-1,
                    ),
                    axis=(-1, -2),
                    dtype=jnp.int16,
                )  # (len(other_team_ids), max_units)
                all_units = all_units.replace(
                    energy=all_units.energy.at[other_team_ids].set(
                        jnp.where(
                            other_team_unit_mask[:, :, None]
                            & (other_units_adjacent_sapped_count[:, :, None] > 0),
                            all_units.energy[other_team_ids]
                            - jnp.array(
                                jnp.array(params.unit_sap_cost, dtype=jnp.float32)
                                * params.unit_sap_dropoff_factor
                                * other_units_adjacent_sapped_count[:, :, None].astype(jnp.float32),
                                dtype=jnp.int16,
                            ),
                            all_units.energy[other_team_ids],
                        )
                    )
                )

                # remove unit_sap_cost energy from units that tried to sap some position within the unit's range
                all_units = all_units.replace(
                    energy=all_units.energy.at[t].set(
                        jnp.where(
                            team_unit_sapped[:, None],
                            all_units.energy[t] - params.unit_sap_cost,
                            all_units.energy[t],
                        )
                    )
                )
            return all_units

        state = state.replace(
            units=sap_unit(
                original_unit_energy,
                state.units,
                sap_action_mask,
                sap_action_deltas,
                state.units_mask,
            )
        )

        """resolve collisions and energy void fields"""

        # compute energy void fields for all teams and the energy + unit counts
        unit_aggregate_energy_void_map = jnp.zeros(
            shape=(self.fixed_env_params.num_teams, self.fixed_env_params.map_width, self.fixed_env_params.map_height),
            dtype=jnp.int16,
        )
        unit_counts_map = self.compute_unit_counts_map(state, params)
        unit_aggregate_energy_map = jnp.zeros(
            shape=(self.fixed_env_params.num_teams, self.fixed_env_params.map_width, self.fixed_env_params.map_height),
            dtype=jnp.int16,
        )
        for t in range(self.fixed_env_params.num_teams):

            def scan_body(carry, x):
                agg_energy_void_map, agg_energy_map = carry
                unit_energy, unit_position, unit_mask = x
                agg_energy_map = agg_energy_map.at[
                    unit_position[0], unit_position[1]
                ].add(unit_energy[0] * unit_mask.astype(jnp.int16))
                for deltas in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_pos = unit_position + jnp.array(deltas, dtype=jnp.int16)
                    in_map = (
                        (new_pos[0] >= 0)
                        & (new_pos[0] < self.fixed_env_params.map_width)
                        & (new_pos[1] >= 0)
                        & (new_pos[1] < self.fixed_env_params.map_height)
                    )
                    agg_energy_void_map = agg_energy_void_map.at[
                        new_pos[0], new_pos[1]
                    ].add(unit_energy[0] * unit_mask.astype(jnp.int16) * in_map.astype(jnp.int16))
                return (agg_energy_void_map, agg_energy_map), None

            agg_energy_void_map, agg_energy_map = jax.lax.scan(
                scan_body,
                (unit_aggregate_energy_void_map[t], unit_aggregate_energy_map[t]),
                (original_unit_energy[t], state.units.position[t], state.units_mask[t]),
            )[0]
            unit_aggregate_energy_void_map = unit_aggregate_energy_void_map.at[t].add(
                agg_energy_void_map
            )
            unit_aggregate_energy_map = unit_aggregate_energy_map.at[t].add(
                agg_energy_map
            )

        # resolve collisions and keep only the surviving units
        for t in range(self.fixed_env_params.num_teams):
            other_team_ids = jnp.array(
                [t2 for t2 in range(self.fixed_env_params.num_teams) if t2 != t]
            )
            # get the energy map for the current team
            opposing_unit_counts_map = unit_counts_map[other_team_ids].sum(
                axis=0
            )  # (map_width, map_height)
            team_energy_map = unit_aggregate_energy_map[t]
            opposing_aggregate_energy_map = unit_aggregate_energy_map[
                other_team_ids
            ].max(
                axis=0
            )  # (map_width, map_height)
            # unit survives if there are opposing units on the tile, and if the opposing unit stack has less energy on the tile than the current unit
            surviving_unit_mask = jax.vmap(
                lambda unit_position: (
                    opposing_unit_counts_map[unit_position[0], unit_position[1]] == 0
                )
                | (
                    opposing_aggregate_energy_map[unit_position[0], unit_position[1]]
                    < team_energy_map[unit_position[0], unit_position[1]]
                )
            )(state.units.position[t])
            state = state.replace(
                units_mask=state.units_mask.at[t].set(
                    surviving_unit_mask & state.units_mask[t]
                )
            )
        # apply energy void fields
        for t in range(self.fixed_env_params.num_teams):
            other_team_ids = jnp.array(
                [t2 for t2 in range(self.fixed_env_params.num_teams) if t2 != t]
            )
            oppposition_energy_void_map = unit_aggregate_energy_void_map[
                other_team_ids
            ].sum(
                axis=0
            )  # (map_width, map_height)
            # unit on team t loses energy to void field equal to params.unit_energy_void_factor * void_energy / num units stacked with unit on the same tile
            team_unit_energy = state.units.energy[t] - jnp.floor(
                jax.vmap(
                    lambda unit_position: params.unit_energy_void_factor
                    * oppposition_energy_void_map[unit_position[0], unit_position[1]].astype(jnp.float32)
                    / unit_counts_map[t][unit_position[0], unit_position[1]].astype(jnp.float32)
                )(state.units.position[t])[..., None]
            ).astype(jnp.int16)
            state = state.replace(
                units=state.units.replace(
                    energy=state.units.energy.at[t].set(team_unit_energy)
                )
            )

        """apply energy field to the units"""

        # Update unit energy based on the energy field and nebula tileof their current position
        def update_unit_energy(unit: UnitState, mask):
            x, y = unit.position
            energy_gain = (
                state.map_features.energy[x, y]
                - (state.map_features.tile_type[x, y] == NEBULA_TILE).astype(jnp.int16)
                * params.nebula_tile_energy_reduction
            )
            # if energy gain is less than 0
            # new_energy = jnp.where((unit.energy < 0) & (energy_gain < 0))
            new_energy = jnp.clip(
                unit.energy + energy_gain,
                params.min_unit_energy,
                params.max_unit_energy,
            )
            # if unit already had negative energy due to opposition units and after energy field/nebula tile it is still below 0, then it will be removed next step
            # and we keep its energy value at whatever it is
            new_energy = jnp.where(
                (unit.energy < 0) & (unit.energy + energy_gain < 0),
                unit.energy,
                new_energy,
            )
            return UnitState(
                position=unit.position, energy=jnp.where(mask, new_energy, unit.energy)
            )

        # Apply the energy update for all units of both teams
        state = state.replace(
            units=jax.vmap(
                lambda team_units, team_mask: jax.vmap(update_unit_energy)(
                    team_units, team_mask
                )
            )(state.units, state.units_mask)
        )

        """spawn new units in"""
        spawn_units_in = state.match_steps % params.spawn_rate == 0

        # TODO (stao): only logic in code that probably doesn't not handle more than 2 teams, everything else is vmapped across teams
        def spawn_team_units(state: EnvState):
            team_0_unit_count = state.units_mask[0].sum()
            team_1_unit_count = state.units_mask[1].sum()
            team_0_new_unit_id = state.units_mask[0].argmin()
            team_1_new_unit_id = state.units_mask[1].argmin()
            state = state.replace(
                units=state.units.replace(
                    position=jnp.where(
                        team_0_unit_count < params.max_units,
                        state.units.position.at[0, team_0_new_unit_id, :].set(
                            jnp.array([0, 0], dtype=jnp.int16)
                        ),
                        state.units.position,
                    )
                )
            )
            state = state.replace(
                units=state.units.replace(
                    energy=jnp.where(
                        team_0_unit_count < params.max_units,
                        state.units.energy.at[0, team_0_new_unit_id, :].set(
                            jnp.array([params.init_unit_energy], dtype=jnp.int16)
                        ),
                        state.units.energy,
                    )
                )
            )
            state = state.replace(
                units=state.units.replace(
                    position=jnp.where(
                        team_1_unit_count < params.max_units,
                        state.units.position.at[1, team_1_new_unit_id, :].set(
                            jnp.array(
                                [params.map_width - 1, params.map_height - 1],
                                dtype=jnp.int16,
                            )
                        ),
                        state.units.position,
                    )
                )
            )
            state = state.replace(
                units=state.units.replace(
                    energy=jnp.where(
                        team_1_unit_count < params.max_units,
                        state.units.energy.at[1, team_1_new_unit_id, :].set(
                            jnp.array([params.init_unit_energy], dtype=jnp.int16)
                        ),
                        state.units.energy,
                    )
                )
            )
            state = state.replace(
                units_mask=state.units_mask.at[0, team_0_new_unit_id].set(
                    jnp.where(
                        team_0_unit_count < params.max_units,
                        True,
                        state.units_mask[0, team_0_new_unit_id],
                    )
                )
            )
            state = state.replace(
                units_mask=state.units_mask.at[1, team_1_new_unit_id].set(
                    jnp.where(
                        team_1_unit_count < params.max_units,
                        True,
                        state.units_mask[1, team_1_new_unit_id],
                    )
                )
            )
            # state = jnp.where(team_0_unit_count < params.max_units, spawn_unit(state, 0, team_0_new_unit_id, [0, 0], params), state)
            # state = jnp.where(team_1_unit_count < params.max_units, spawn_unit(state, 1, team_1_new_unit_id, [params.map_width - 1, params.map_height - 1], params), state)
            return state

        state = jax.lax.cond(
            spawn_units_in, lambda: spawn_team_units(state), lambda: state
        )

        state = self.compute_sensor_masks(state, params)

        # Shift objects around in space
        # Move the nebula tiles in state.map_features.tile_types up by 1 and to the right by 1
        # this is also symmetric nebula tile movement
        new_tile_types_map = jnp.roll(
            state.map_features.tile_type,
            shift=(
                1 * jnp.sign(params.nebula_tile_drift_speed),
                -1 * jnp.sign(params.nebula_tile_drift_speed),
            ),
            axis=(0, 1),
        )
        new_tile_types_map = jnp.where(
            state.steps * params.nebula_tile_drift_speed % 1 == 0,
            new_tile_types_map,
            state.map_features.tile_type,
        )
        # new_energy_nodes = state.energy_nodes + jnp.array([1 * jnp.sign(params.energy_node_drift_speed), -1 * jnp.sign(params.energy_node_drift_speed)])

        energy_node_deltas = jnp.round(
            jax.random.uniform(
                key=key,
                shape=(self.fixed_env_params.max_energy_nodes // 2, 2),
                minval=-params.energy_node_drift_magnitude,
                maxval=params.energy_node_drift_magnitude,
            )
        ).astype(jnp.int16)
        energy_node_deltas_symmetric = jnp.stack(
            [-energy_node_deltas[:, 1], -energy_node_deltas[:, 0]], axis=-1
        )
        # TODO symmetric movement
        # energy_node_deltas = jnp.round(jax.random.uniform(key=key, shape=(params.max_energy_nodes // 2, 2), minval=-params.energy_node_drift_magnitude, maxval=params.energy_node_drift_magnitude)).astype(jnp.int16)
        energy_node_deltas = jnp.concatenate(
            (energy_node_deltas, energy_node_deltas_symmetric)
        )
        new_energy_nodes = jnp.clip(
            state.energy_nodes + energy_node_deltas,
            min=jnp.array([0, 0], dtype=jnp.int16),
            max=jnp.array(
                [self.fixed_env_params.map_width - 1, self.fixed_env_params.map_height - 1],
                dtype=jnp.int16
            ),
        )
        new_energy_nodes = jnp.where(
            state.steps * params.energy_node_drift_speed % 1 == 0,
            new_energy_nodes,
            state.energy_nodes,
        )
        state = state.replace(
            map_features=state.map_features.replace(tile_type=new_tile_types_map),
            energy_nodes=new_energy_nodes,
        )

        # Compute relic scores
        def team_relic_score(unit_counts_map):
            scores = (unit_counts_map > 0) & (state.relic_nodes_map_weights > 0)
            return jnp.sum(scores, dtype=jnp.int32)

        # note we need to recompue unit counts since units can get removed due to collisions
        team_scores = jax.vmap(team_relic_score)(
            self.compute_unit_counts_map(state, params, exclude_negative_energy_units=True)
        )
        # Update team points
        state = state.replace(team_points=state.team_points + team_scores)

        # if match ended, then remove all units, update team wins, reset team points
        winner_by_points = jnp.where(
            state.team_points.max() > state.team_points.min(),
            jnp.argmax(state.team_points),
            -1,
        )
        winner_by_energy = jnp.sum(
            state.units.energy[..., 0] * state.units_mask.astype(jnp.int16), axis=1
        )
        winner_by_energy = jnp.where(
            winner_by_energy.max() > winner_by_energy.min(),
            jnp.argmax(winner_by_energy),
            -1,
        )

        winner = jnp.where(
            winner_by_points != -1,
            winner_by_points,
            jnp.where(
                winner_by_energy != -1,
                winner_by_energy,
                jax.random.randint(key, shape=(), minval=0, maxval=params.num_teams),
            ),
        )
        match_ended = state.match_steps >= params.max_steps_in_match

        state = state.replace(
            match_steps=jnp.where(match_ended, -1, state.match_steps),
            team_points=jnp.where(
                match_ended, jnp.zeros_like(state.team_points), state.team_points
            ),
            team_wins=jnp.where(
                match_ended, state.team_wins.at[winner].add(1), state.team_wins
            ),
        )
        # Update state's step count
        state = state.replace(steps=state.steps + 1, match_steps=state.match_steps + 1)
        truncated = (
            state.steps
            >= (params.max_steps_in_match + 1) * params.match_count_per_episode
        )
        reward = dict()
        for k in range(self.fixed_env_params.num_teams):
            reward[f"player_{k}"] = state.team_wins[k]
        terminated = self.is_terminal(state, params)
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

        state = gen_state(
            key=key,
            env_params=params,
            max_units=self.fixed_env_params.max_units,
            num_teams=self.fixed_env_params.num_teams,
            map_type=self.fixed_env_params.map_type,
            map_width=self.fixed_env_params.map_width,
            map_height=self.fixed_env_params.map_height,
            max_energy_nodes=self.fixed_env_params.max_energy_nodes,
            max_relic_nodes=self.fixed_env_params.max_relic_nodes,
            relic_config_size=self.fixed_env_params.relic_config_size,
        )
        state = self.compute_energy_features(state, params)
        state = self.compute_sensor_masks(state, params)

        return self.get_obs(state, params=params, key=key), state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, truncated, info = self.step_env(
            key, state, action, params
        )
        info["final_state"] = state_st
        info["final_observation"] = obs_st
        done = terminated | truncated
        
        if self.auto_reset:
            obs_re, state_re = self.reset_env(key_reset, params)
            # Use lax.cond to efficiently choose between obs_re and obs_st
            obs = jax.lax.cond(
                done,
                lambda: obs_re,
                lambda: obs_st
            )
            state = jax.lax.cond(
                done,
                lambda: state_re,
                lambda: state_st
            )
        else:
            obs = obs_st
            state = state_st

        # all agents terminate/truncate at same time
        terminated_dict = dict()
        truncated_dict = dict()
        for k in range(self.fixed_env_params.num_teams):
            terminated_dict[f"player_{k}"] = terminated
            truncated_dict[f"player_{k}"] = truncated
            info[f"player_{k}"] = dict()
        return obs, state, reward, terminated_dict, truncated_dict, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params

        obs, state = self.reset_env(key, params)
        return obs, state

    # @functools.partial(jax.jit, static_argnums=(0, 2))
    def get_obs(self, state: EnvState, params=None, key=None) -> EnvObs:
        """Return observation from raw state, handling partial observability."""
        obs = dict()

        def update_unit_mask(unit_position, unit_mask, sensor_mask):
            return unit_mask & sensor_mask[unit_position[0], unit_position[1]]

        def update_team_unit_mask(unit_position, unit_mask, sensor_mask):
            return jax.vmap(update_unit_mask, in_axes=(0, 0, None))(
                unit_position, unit_mask, sensor_mask
            )

        def update_relic_nodes_mask(relic_nodes_mask, relic_nodes, sensor_mask):
            return jax.vmap(
                lambda r_mask, r, s_mask: r_mask & s_mask[r[0], r[1]],
                in_axes=(0, 0, None),
            )(relic_nodes_mask, relic_nodes, sensor_mask)

        for t in range(self.fixed_env_params.num_teams):
            other_team_ids = jnp.array(
                [t2 for t2 in range(self.fixed_env_params.num_teams) if t2 != t]
            )
            new_unit_masks = jax.vmap(update_team_unit_mask, in_axes=(0, 0, None))(
                state.units.position[other_team_ids],
                state.units_mask[other_team_ids],
                state.sensor_mask[t],
            )
            new_unit_masks = state.units_mask.at[other_team_ids].set(new_unit_masks)

            new_relic_nodes_mask = update_relic_nodes_mask(
                state.relic_nodes_mask, state.relic_nodes, state.sensor_mask[t]
            )
            team_obs = EnvObs(
                units=UnitState(
                    position=jnp.where(
                        new_unit_masks[..., None], state.units.position, -1
                    ),
                    energy=jnp.where(new_unit_masks[..., None], state.units.energy, -1)[
                        ..., 0
                    ],
                ),
                units_mask=new_unit_masks,
                sensor_mask=state.sensor_mask[t],
                map_features=MapTile(
                    energy=jnp.where(
                        state.sensor_mask[t], state.map_features.energy, -1
                    ),
                    tile_type=jnp.where(
                        state.sensor_mask[t], state.map_features.tile_type, -1
                    ),
                ),
                team_points=state.team_points,
                team_wins=state.team_wins,
                steps=state.steps,
                match_steps=state.match_steps,
                relic_nodes=jnp.where(
                    new_relic_nodes_mask[..., None], state.relic_nodes, -1
                ),
                relic_nodes_mask=new_relic_nodes_mask,
            )
            obs[f"player_{t}"] = team_obs
        return obs

    @functools.partial(jax.jit, static_argnums=(0, ))
    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal. This never occurs. Game is only done when the time limit is reached."""
        terminated = jnp.array(False)
        return terminated

    @property
    def name(self) -> str:
        """Environment name."""
        return "Lux AI Season 3"

    def render(self, state: EnvState, params: EnvParams):
        self.renderer.render(state, params)

    def action_space(self, params: Optional[EnvParams] = None):
        """Action space of the environment."""
        low = np.zeros((self.fixed_env_params.max_units, 3))
        low[:, 1:] = -env_params_ranges["unit_sap_range"][-1]
        high = np.ones((self.fixed_env_params.max_units, 3)) * 6
        high[:, 1:] = env_params_ranges["unit_sap_range"][-1]
        return spaces.Dict(
            dict(player_0=MultiDiscrete(low, high), player_1=MultiDiscrete(low, high))
        )

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(10)

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Discrete(10)


