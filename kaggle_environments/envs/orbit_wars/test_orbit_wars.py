import math
import random
import unittest
from types import SimpleNamespace
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    interpreter,
    generate_planets,
    distance,
    CENTER,
    ROTATION_RADIUS_LIMIT,
)


class TestOrbitWars(unittest.TestCase):

    def _advance_step_counter(self, state):
        next_step = getattr(state[0].observation, "step", 0) + 1
        for agent_state in state:
            agent_state.observation.step = next_step
        return state

    def test_symmetry(self):
        # Mock state for step 0
        state = [
            SimpleNamespace(
                observation=SimpleNamespace(step=0),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
            SimpleNamespace(
                observation=SimpleNamespace(player=1),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
        ]
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=5, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        obs = new_state[0].observation
        planets = obs.planets

        self.assertTrue(len(planets) >= 4)
        self.assertEqual(len(planets) % 4, 0)

        # Check symmetry across center (50,50)
        for i in range(0, len(planets), 4):
            p0 = planets[i]
            p3 = planets[i + 3]

            self.assertTrue(math.isclose(p0[2] + p3[2], 100.0, abs_tol=1e-5))
            self.assertTrue(math.isclose(p0[3] + p3[3], 100.0, abs_tol=1e-5))
            self.assertEqual(p0[4], p3[4])

    def test_combat_resolution_user_example(self):
        # Planet format: [id, owner, x, y, radius, ships, production]
        # Fleet format:  [id, owner, x, y, angle, from_planet_id, ships]
        # Mock state for 4 players
        state = [
            SimpleNamespace(
                observation=SimpleNamespace(
                    step=1,
                    planets=[[0, -1, 80, 80, 5, 10, 0]],
                    fleets=[],
                    next_fleet_id=0,
                    angular_velocity=0.01,
                    initial_planets=[[0, -1, 80, 80, 5, 10, 0]],
                    comets=[],
                    comet_planet_ids=[],
                ),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
            SimpleNamespace(
                observation=SimpleNamespace(player=1),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
            SimpleNamespace(
                observation=SimpleNamespace(player=2),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
            SimpleNamespace(
                observation=SimpleNamespace(player=3),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
        ]
        # Fleets placed slightly before planet 0 (at 80,80) so they move into it
        state[0].observation.fleets = [
            [0, 0, 76.0, 80.0, 0.0, 1, 41],  # P0: 41
            [1, 1, 76.0, 80.0, 0.0, 2, 20],  # P1: 20
            [2, 1, 76.0, 80.0, 0.0, 2, 20],  # P1: 20 (Total 40)
            [3, 2, 76.0, 80.0, 0.0, 3, 42],  # P2: 42
        ]

        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=5, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)

        planets = new_state[0].observation.planets
        self.assertEqual(planets[0][1], -1)
        self.assertEqual(planets[0][5], 9)

    def test_4_player_initialization(self):
        # Mock state for 4 players
        state = [
            SimpleNamespace(
                observation=SimpleNamespace(step=0),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
            SimpleNamespace(
                observation=SimpleNamespace(player=1),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
            SimpleNamespace(
                observation=SimpleNamespace(player=2),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
            SimpleNamespace(
                observation=SimpleNamespace(player=3),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
        ]
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=5, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        obs = new_state[0].observation
        planets = obs.planets

        # Check that 4 planets are owned by players 0, 1, 2, 3
        owned = [p for p in planets if p[1] != -1]
        self.assertEqual(len(owned), 4)
        owners = set(p[1] for p in owned)
        self.assertEqual(owners, {0, 1, 2, 3})

    def test_comet_spawn_keeps_initial_planets_synced_across_players(self):
        # Comet spawning is RNG-driven; seed so the 49-step window is
        # deterministic and reliably contains a spawn.
        random.seed(0)
        state = [
            SimpleNamespace(
                observation=SimpleNamespace(step=0),
                action=[],
                status="ACTIVE",
                reward=0,
            )
        ] + [
            SimpleNamespace(
                observation=SimpleNamespace(player=i),
                action=[],
                status="ACTIVE",
                reward=0,
            )
            for i in range(1, 4)
        ]
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=5, episodeSteps=120, cometSpeed=4),
            done=False,
        )

        state = interpreter(state, env)
        state = self._advance_step_counter(state)

        for _ in range(49):
            for agent_state in state:
                agent_state.action = []
            state = interpreter(state, env)
            state = self._advance_step_counter(state)

        obs0 = state[0].observation

        self.assertTrue(obs0.comets)
        self.assertEqual(len(obs0.initial_planets), len(obs0.planets))
        for other_state in state[1:]:
            other_obs = other_state.observation
            self.assertEqual(obs0.comet_planet_ids, other_obs.comet_planet_ids)
            self.assertEqual(obs0.initial_planets, other_obs.initial_planets)

    def test_4p_home_planets_rotationally_symmetric(self):
        # In 4p, the 4 home planets must be a 4-fold rotationally symmetric
        # set about the board center: rotating any home position by 90° CCW
        # must land on another home position. This guarantees every player's
        # view is identical (just rotated), so all are equidistant from
        # the planet rotationally ahead and behind.
        for _ in range(50):
            state = [
                SimpleNamespace(
                    observation=SimpleNamespace(step=0),
                    action=[],
                    status="ACTIVE",
                    reward=0,
                ),
            ] + [
                SimpleNamespace(
                    observation=SimpleNamespace(player=i),
                    action=[],
                    status="ACTIVE",
                    reward=0,
                )
                for i in range(1, 4)
            ]
            env = SimpleNamespace(
                configuration=SimpleNamespace(shipSpeed=5, episodeSteps=500, cometSpeed=4),
                done=False,
            )
            new_state = interpreter(state, env)
            owned = [p for p in new_state[0].observation.planets if p[1] != -1]
            self.assertEqual(len(owned), 4)
            positions = [(p[2], p[3]) for p in owned]
            # Rotate each position 90° CCW about CENTER and verify it lands on
            # another home position.
            for x, y in positions:
                rx = CENTER - (y - CENTER)
                ry = CENTER + (x - CENTER)
                self.assertTrue(
                    any(
                        math.isclose(rx, px, abs_tol=1e-5)
                        and math.isclose(ry, py, abs_tol=1e-5)
                        for px, py in positions
                    ),
                    msg=f"home set not 4-fold rotational: {positions}",
                )

    def _make_state(self, planets, fleets, step=1):
        """Helper to build a minimal 2-player state for testing."""
        return [
            SimpleNamespace(
                observation=SimpleNamespace(
                    step=step,
                    planets=planets,
                    fleets=fleets,
                    next_fleet_id=100,
                    angular_velocity=0.01,
                    initial_planets=[p[:] for p in planets],
                    comets=[],
                    comet_planet_ids=[],
                ),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
            SimpleNamespace(
                observation=SimpleNamespace(player=1),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
        ]

    def _make_state_4p(self, planets, fleets, step=1):
        """Helper to build a minimal 4-player state for testing."""
        return [
            SimpleNamespace(
                observation=SimpleNamespace(
                    step=step,
                    planets=planets,
                    fleets=fleets,
                    next_fleet_id=100,
                    angular_velocity=0.01,
                    initial_planets=[p[:] for p in planets],
                    comets=[],
                    comet_planet_ids=[],
                ),
                action=[],
                status="ACTIVE",
                reward=0,
            ),
        ] + [
            SimpleNamespace(
                observation=SimpleNamespace(player=i),
                action=[],
                status="ACTIVE",
                reward=0,
            )
            for i in range(1, 4)
        ]

    def test_rewards_set_at_max_steps(self):
        # When the game reaches episodeSteps without elimination,
        # rewards should reflect each player's total ships.
        # The framework sets obs.step = len(steps) AFTER the interpreter runs,
        # so the interpreter sees step = episodeSteps - 2 on the final call.
        planets = [
            [0, 0, 80, 80, 3, 50, 1],
            [1, 1, 20, 20, 3, 30, 1],
        ]
        # Use step = episodeSteps - 2 (the last step the interpreter sees)
        state = self._make_state(planets, [], step=498)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        # Player 0 has more ships (51 vs 31), so wins
        self.assertEqual(new_state[0].reward, 1)
        self.assertEqual(new_state[1].reward, -1)
        self.assertEqual(new_state[0].status, "DONE")

    def test_reward_elimination_winner_and_loser(self):
        # Player 0 has planets, player 1 has nothing -> elimination
        planets = [
            [0, 0, 80, 80, 3, 50, 1],
        ]
        state = self._make_state(planets, [])
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        self.assertEqual(new_state[0].reward, 1)
        self.assertEqual(new_state[1].reward, -1)
        self.assertEqual(new_state[0].status, "DONE")
        self.assertEqual(new_state[1].status, "DONE")

    def test_reward_elimination_via_fleets_only(self):
        # Player 1 has no planets but has a fleet -> not eliminated yet
        planets = [
            [0, 0, 80, 80, 3, 50, 1],
        ]
        fleets = [
            [0, 1, 30, 30, 0.0, 99, 10],
        ]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        # Player 1 still has a fleet, game continues
        self.assertEqual(new_state[0].status, "ACTIVE")
        self.assertEqual(new_state[0].reward, 0)

    def test_reward_tie_at_max_steps(self):
        # Both players have equal ships at game end -> both get 1
        planets = [
            [0, 0, 80, 80, 3, 30, 1],
            [1, 1, 20, 20, 3, 30, 1],
        ]
        state = self._make_state(planets, [], step=498)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        # Both have 31 ships (30 + 1 production), both win
        self.assertEqual(new_state[0].reward, 1)
        self.assertEqual(new_state[1].reward, 1)

    def test_reward_all_eliminated(self):
        # No players have planets or fleets -> all lose
        planets = [
            [0, -1, 80, 80, 3, 50, 1],
        ]
        state = self._make_state(planets, [])
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        # Neither player is alive, both get -1
        self.assertEqual(new_state[0].reward, -1)
        self.assertEqual(new_state[1].reward, -1)

    def test_reward_4_player_elimination(self):
        # Only player 2 survives
        planets = [
            [0, 2, 80, 80, 3, 40, 1],
        ]
        state = self._make_state_4p(planets, [])
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        self.assertEqual(new_state[0].reward, -1)
        self.assertEqual(new_state[1].reward, -1)
        self.assertEqual(new_state[2].reward, 1)
        self.assertEqual(new_state[3].reward, -1)

    def test_reward_includes_fleet_ships(self):
        # Player 0 has fewer planet ships but more fleet ships
        planets = [
            [0, 0, 80, 80, 3, 10, 1],
            [1, 1, 20, 20, 3, 30, 1],
        ]
        fleets = [
            [0, 0, 50, 30, 0.0, 0, 50],  # P0 fleet with 50 ships
        ]
        state = self._make_state(planets, fleets, step=498)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        # P0: 11 planet + 50 fleet = 61, P1: 31 planet = 31
        self.assertEqual(new_state[0].reward, 1)
        self.assertEqual(new_state[1].reward, -1)

    def test_fleet_removed_when_hitting_sun(self):
        # Planet far from sun, fleet aimed directly at the sun
        planets = [[0, 0, 80, 50, 3, 50, 1]]
        # Fleet heading left toward sun center (angle = pi)
        fleets = [[0, 0, 60, 50, math.pi, 0, 10]]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        remaining = new_state[0].observation.fleets
        self.assertEqual(len(remaining), 0)

    def test_fleet_removed_when_leaving_board(self):
        # Fleet near the right edge heading right (angle = 0)
        planets = [[0, 0, 80, 50, 3, 50, 1]]
        fleets = [[0, 0, 99.5, 50, 0.0, 0, 10]]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        remaining = new_state[0].observation.fleets
        self.assertEqual(len(remaining), 0)

    def test_fleet_survives_inside_board(self):
        # Fleet in middle of board heading right, should survive
        planets = [[0, 0, 80, 80, 3, 50, 1]]
        fleets = [[0, 0, 50, 30, 0.0, 0, 10]]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        remaining = new_state[0].observation.fleets
        self.assertEqual(len(remaining), 1)

    def test_fast_fleet_hits_planet_before_leaving_board(self):
        # A fast fleet whose move ends out-of-bounds should still resolve
        # combat at any planet its segment crosses on the way out.
        # Planet at (98, 50) radius 2 sits beyond r=50 from center so it
        # does not rotate (r + radius = 50, not < ROTATION_RADIUS_LIMIT).
        # Fleet has 1000 ships -> max speed 6. It moves from x=95 to x=101
        # (out of bounds) but its segment passes through the planet.
        planets = [[0, 1, 98.0, 50.0, 2.0, 50, 1]]
        fleets = [[0, 0, 95.0, 50.0, 0.0, 99, 1000]]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        # Combat should have resolved: attacker (1000) vs garrison (50+1 prod = 51)
        self.assertEqual(p[1], 0)
        self.assertEqual(p[5], 1000 - 51)
        self.assertEqual(len(new_state[0].observation.fleets), 0)

    def test_fast_fleet_hits_planet_before_sun(self):
        # A fast fleet whose segment crosses both a planet and the sun
        # should resolve combat at the planet rather than be silently
        # consumed by the sun.
        planets = [[0, 1, 62.0, 50.0, 2.0, 50, 1]]
        fleets = [[0, 0, 65.0, 50.0, math.pi, 99, 1000]]
        state = self._make_state(planets, fleets)
        # Disable rotation to keep the planet fixed for this geometry test.
        state[0].observation.angular_velocity = 0.0
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], 0)
        self.assertEqual(p[5], 1000 - 51)
        self.assertEqual(len(new_state[0].observation.fleets), 0)

    def test_combat_simple_capture(self):
        # Fleet of 30 attacks neutral planet with 10 ships
        # Attacker wins with 30-10=20 ships remaining, captures planet
        planets = [[0, -1, 80, 50, 3, 10, 1]]
        fleets = [[0, 0, 76.0, 50.0, 0.0, 99, 30]]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], 0)  # captured by player 0
        self.assertEqual(p[5], 20)  # 30 - 10 = 20

    def test_combat_simple_reinforce(self):
        # Fleet of 25 lands on own planet with 10 ships
        # Ships are added: 10 + 25 = 35 (plus 1 production)
        planets = [[0, 0, 80, 50, 3, 10, 1]]
        fleets = [[0, 0, 76.0, 50.0, 0.0, 99, 25]]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], 0)  # still player 0
        # 10 garrison + 1 production + 25 reinforcement = 36
        self.assertEqual(p[5], 36)

    def test_combat_attacker_insufficient(self):
        # Fleet of 5 attacks neutral planet with 20 ships
        # Planet keeps ownership, garrison reduced to 15
        planets = [[0, -1, 80, 50, 3, 20, 1]]
        fleets = [[0, 0, 76.0, 50.0, 0.0, 99, 5]]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], -1)  # still neutral
        self.assertEqual(p[5], 15)  # 20 - 5 = 15

    def test_combat_two_attackers_winner_captures(self):
        # P0 sends 50, P1 sends 30 to neutral planet with 10 ships
        # Attackers fight first: P0 wins with 50-30=20
        # Then 20 vs planet's 10: P0 captures with 10
        planets = [[0, -1, 80, 50, 3, 10, 1]]
        fleets = [
            [0, 0, 76.0, 50.0, 0.0, 99, 50],
            [1, 1, 76.0, 50.0, 0.0, 99, 30],
        ]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], 0)  # P0 captures
        self.assertEqual(p[5], 10)  # 20 survivors - 10 garrison = 10

    def test_combat_two_attackers_tie_all_destroyed(self):
        # P0 sends 30, P1 sends 30 to neutral planet with 10 ships
        # Tie: all attacking ships destroyed, planet unchanged
        planets = [[0, -1, 80, 50, 3, 10, 1]]
        fleets = [
            [0, 0, 76.0, 50.0, 0.0, 99, 30],
            [1, 1, 76.0, 50.0, 0.0, 99, 30],
        ]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], -1)  # still neutral
        self.assertEqual(p[5], 10)  # unchanged

    def test_combat_winner_reinforces_own_planet(self):
        # Planet owned by P0 with 15 ships
        # P0 sends 40 (reinforce), P1 sends 25 (attack)
        # Attackers fight: P0 wins with 40-25=15 survivors
        # P0 owns planet, so 15 survivors reinforce: 15 + 15 + 1 prod = 31
        planets = [[0, 0, 80, 50, 3, 15, 1]]
        fleets = [
            [0, 0, 76.0, 50.0, 0.0, 99, 40],
            [1, 1, 76.0, 50.0, 0.0, 99, 25],
        ]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], 0)  # still P0
        # 15 garrison + 1 production + 15 survivors = 31
        self.assertEqual(p[5], 31)

    def test_combat_winner_attacks_enemy_planet(self):
        # Planet owned by P1 with 5 ships
        # P0 sends 50, P2 sends 20
        # Attackers fight: P0 wins with 50-20=30
        # 30 vs P1's garrison (5 + 1 prod = 6): P0 captures with 30-6=24
        planets = [[0, 1, 80, 50, 3, 5, 1]]
        fleets = [
            [0, 0, 76.0, 50.0, 0.0, 99, 50],
            [1, 2, 76.0, 50.0, 0.0, 99, 20],
        ]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], 0)  # P0 captures
        # garrison was 5+1prod=6, attacker has 30: 30-6=24
        self.assertEqual(p[5], 24)

    def test_combat_multiple_fleets_same_owner(self):
        # Two fleets from P0 (20 + 15 = 35) attack neutral with 10
        # P0 captures with 35-10=25
        planets = [[0, -1, 80, 50, 3, 10, 1]]
        fleets = [
            [0, 0, 76.0, 50.0, 0.0, 99, 20],
            [1, 0, 76.0, 50.0, 0.0, 99, 15],
        ]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500, cometSpeed=4),
            done=False,
        )

        new_state = interpreter(state, env)
        p = new_state[0].observation.planets[0]
        self.assertEqual(p[1], 0)  # P0 captures
        self.assertEqual(p[5], 25)  # 35 - 10 = 25


    def test_seed_hidden_from_agents_but_in_replay(self):
        # The seed drives planet placement and the comet schedule (which is
        # supposed to be hidden info). Agents must NOT see the seed via the
        # configuration they're handed, but the replay/env must still record
        # it for reproducibility.
        from kaggle_environments import make

        seen_seeds = []
        seen_configs = []

        def spy_agent(observation, configuration):
            seen_seeds.append(configuration.get("seed"))
            seen_configs.append(dict(configuration))
            return []

        chosen_seed = 1234567
        env = make(
            "orbit_wars",
            configuration={"seed": chosen_seed, "episodeSteps": 60},
            debug=True,
        )
        env.run([spy_agent, spy_agent])

        self.assertTrue(seen_seeds, "spy agent never received configuration")
        for s, cfg in zip(seen_seeds, seen_configs):
            self.assertIsNone(
                s, msg=f"agent saw seed={s} in configuration: {cfg}"
            )

        # Seed must still be persisted on the env / replay for reproducibility.
        self.assertEqual(env.info.get("seed"), chosen_seed)
        replay = env.toJSON()
        self.assertEqual(replay["info"].get("seed"), chosen_seed)
        self.assertIsNone(replay["configuration"].get("seed"))

    def test_seed_hidden_when_unset_by_user(self):
        # When the user doesn't supply a seed, the interpreter generates one.
        # That generated seed must also stay out of the agent's configuration
        # view but be recorded in env.info.
        from kaggle_environments import make

        seen_seeds = []

        def spy_agent(observation, configuration):
            seen_seeds.append(configuration.get("seed"))
            return []

        env = make("orbit_wars", configuration={"episodeSteps": 60}, debug=True)
        env.run([spy_agent, spy_agent])

        for s in seen_seeds:
            self.assertIsNone(s, msg=f"agent saw generated seed={s}")
        self.assertIsNotNone(env.info.get("seed"))


if __name__ == "__main__":
    unittest.main()
