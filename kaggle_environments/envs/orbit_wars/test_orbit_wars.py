import unittest
import math
from types import SimpleNamespace
from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter


class TestOrbitWars(unittest.TestCase):

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
            configuration=SimpleNamespace(shipSpeed=5, episodeSteps=500), done=False
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
            configuration=SimpleNamespace(shipSpeed=5, episodeSteps=500), done=False
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
            configuration=SimpleNamespace(shipSpeed=5, episodeSteps=500), done=False
        )

        new_state = interpreter(state, env)
        obs = new_state[0].observation
        planets = obs.planets

        # Check that 4 planets are owned by players 0, 1, 2, 3
        owned = [p for p in planets if p[1] != -1]
        self.assertEqual(len(owned), 4)
        owners = set(p[1] for p in owned)
        self.assertEqual(owners, {0, 1, 2, 3})

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

    def test_fleet_removed_when_hitting_sun(self):
        # Planet far from sun, fleet aimed directly at the sun
        planets = [[0, 0, 80, 50, 3, 50, 1]]
        # Fleet heading left toward sun center (angle = pi)
        fleets = [[0, 0, 60, 50, math.pi, 0, 10]]
        state = self._make_state(planets, fleets)
        env = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500), done=False
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
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500), done=False
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
            configuration=SimpleNamespace(shipSpeed=6, episodeSteps=500), done=False
        )

        new_state = interpreter(state, env)
        remaining = new_state[0].observation.fleets
        self.assertEqual(len(remaining), 1)


if __name__ == "__main__":
    unittest.main()
