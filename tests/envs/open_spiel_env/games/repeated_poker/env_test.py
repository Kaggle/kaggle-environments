"""Env-level tests for open_spiel_repeated_poker."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env

TEST_REPEATED_POKER_GAME_STRING = open_spiel_env.DEFAULT_REPEATED_POKER_GAME_STRING.replace(
    "calcOddsNumSims=1000000",
    "calcOddsNumSims=1",
)


class RepeatedPokerEnvTest(absltest.TestCase):
    def test_default_repeated_poker(self):
        env = make("open_spiel_repeated_poker", {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING})
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Default repeated_poker now includes hand odds calculations which take
        # ~1-2s per action, so we avoid stepping until the end of the game.
        for i in range(2):
            if i % 2 == 0:
                env.step([{"submission": -1}, {"submission": 0}])
            else:
                env.step([{"submission": 0}, {"submission": -1}])
        self.assertEqual(len(env.os_state.acpc_hand_histories()), 2)
        state_dict = json.loads(str(env.os_state))
        self.assertTrue("current_universal_poker_json" in state_dict)
        current_hand_dict = json.loads(state_dict["current_universal_poker_json"])
        self.assertTrue("odds" in current_hand_dict)
        self.assertEqual(len(current_hand_dict["odds"]), 4)

    def test_poker_set_num_hands(self):
        num_hands = 2
        config = {
            "openSpielGameString": TEST_REPEATED_POKER_GAME_STRING,
            "setNumHands": num_hands,
        }
        env = make(
            "open_spiel_repeated_poker",
            config,
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for i in range(num_hands):
            if i % 2 == 0:
                env.step([{"submission": -1}, {"submission": 0}])
            else:
                env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [0.0, 0.0])

    def test_repeated_poker_preset_hands_replay(self):
        base_env = make(
            "open_spiel_repeated_poker",
            {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING, "setNumHands": 2},
            debug=True,
        )
        base_env.reset()
        base_env.step([{"submission": -1}, {"submission": -1}])
        for i in range(2):
            if i % 2 == 0:
                base_env.step([{"submission": -1}, {"submission": 0}])
            else:
                base_env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(base_env.done)
        base_history = [int(action) for action in base_env.info["actionHistory"]]
        replay_state = base_env.os_game.new_initial_state()
        preset_hands: list[list[int]] = []
        for action in base_history:
            if replay_state.is_chance_node():
                hand_idx = len(replay_state.acpc_hand_histories())
                while len(preset_hands) <= hand_idx:
                    preset_hands.append([])
                preset_hands[hand_idx].append(action)
            replay_state.apply_action(action)
        if preset_hands and preset_hands[0]:
            preset_hands[0].append(preset_hands[0][-1])
        preset_env = make(
            "open_spiel_repeated_poker",
            {
                "openSpielGameString": TEST_REPEATED_POKER_GAME_STRING,
                "setNumHands": 2,
                "presetHands": [hand[:] for hand in preset_hands],
            },
            debug=True,
        )
        preset_env.reset()
        preset_env.step([{"submission": -1}, {"submission": -1}])
        for i in range(2):
            if i % 2 == 0:
                preset_env.step([{"submission": -1}, {"submission": 0}])
            else:
                preset_env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(preset_env.done)
        self.assertEqual(
            [int(action) for action in preset_env.info["actionHistory"]],
            base_history,
        )
        self.assertEqual(
            preset_env.info["presetHands"],
            [hand[:] for hand in preset_hands],
        )
        self.assertLess(
            preset_env.info["presetHandsState"]["next_index"][0],
            len(preset_env.info["presetHands"][0]),
        )

    def test_repeated_poker_preset_hands_runs_out(self):
        env = make(
            "open_spiel_repeated_poker",
            {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING, "presetHands": [[0]]},
            debug=True,
        )
        env.reset()
        with self.assertRaisesRegex(ValueError, "presetHands"):
            env.step([{"submission": -1}, {"submission": -1}])

    def test_repeated_poker_preset_hands_conflicts_with_use_openings(self):
        env = make(
            "open_spiel_repeated_poker",
            {
                "openSpielGameString": TEST_REPEATED_POKER_GAME_STRING,
                "presetHands": [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
                "useOpenings": True,
            },
            debug=True,
        )
        env.reset()
        with self.assertRaisesRegex(ValueError, "useOpenings"):
            env.step([{"submission": -1}, {"submission": -1}])

    def test_repeated_poker_load_preset_hands_loads_file(self):
        env = make(
            "open_spiel_repeated_poker",
            {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING, "loadPresetHands": True, "seed": 0},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        self.assertIn("presetHands", env.configuration)
        preset_hands = env.configuration["presetHands"]
        self.assertEqual(len(preset_hands), 100)
        self.assertTrue(all(len(hand) == 9 for hand in preset_hands))

    def test_repeated_poker_load_preset_hands_requires_seed(self):
        env = make(
            "open_spiel_repeated_poker",
            {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING, "loadPresetHands": True},
            debug=True,
        )
        env.reset()
        with self.assertRaisesRegex(ValueError, "seed"):
            env.step([{"submission": -1}, {"submission": -1}])

    def test_repeated_poker_load_preset_hands_conflicts_with_manual(self):
        env = make(
            "open_spiel_repeated_poker",
            {
                "openSpielGameString": TEST_REPEATED_POKER_GAME_STRING,
                "loadPresetHands": True,
                "seed": 0,
                "presetHands": [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
            },
            debug=True,
        )
        env.reset()
        with self.assertRaisesRegex(ValueError, "loadPresetHands"):
            env.step([{"submission": -1}, {"submission": -1}])


if __name__ == "__main__":
    absltest.main()
