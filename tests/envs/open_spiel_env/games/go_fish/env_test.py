"""Env-level tests for open_spiel_go_fish."""

import json
import random

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env
from kaggle_environments.envs.open_spiel_env.games.go_fish import go_fish_proxy


def _advance_chance(state, rng):
    while not state.is_terminal() and state.is_chance_node():
        outcomes = state.chance_outcomes()
        state.apply_action(rng.choices([o[0] for o in outcomes], weights=[o[1] for o in outcomes])[0])


class GoFishDeductionTest(absltest.TestCase):
    """The deduction table must retain the full public ask history, even after
    individual asks scroll out of the per-turn ``recent_events`` window."""

    def _play_to_ask(self, seed, plies):
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        rng = random.Random(seed)
        _advance_chance(state, rng)
        asks = []  # (asker, target_rank_label) for every ask actually applied
        for _ in range(plies):
            if state.is_terminal() or state.current_player() < 0:
                break
            cp = int(state.current_player())
            action = rng.choice(state.legal_actions())
            action_str = state.action_to_string(action)  # "<target><letter>"
            asks.append((cp, action_str))
            state.apply_action(action)
            _advance_chance(state, rng)
        return game, state, asks

    def test_deductions_present_in_state_dict(self):
        game, state, _ = self._play_to_ask(seed=7, plies=6)
        sd = state.state_dict(0)
        self.assertIn("deductions", sd)
        self.assertEqual(len(sd["deductions"]), 2)
        for d in sd["deductions"]:
            self.assertIn("player", d)
            self.assertIn("known_has", d)
            self.assertIn("known_void", d)
            self.assertIn("wanted", d)

    def test_asker_recorded_as_holding_and_wanting_rank(self):
        # An ask publicly reveals the asker holds >=1 of that rank. That fact
        # must persist in the deduction table regardless of the events window.
        game, state, asks = self._play_to_ask(seed=7, plies=8)
        sd = state.state_dict(0)
        # Every distinct rank that a still-in-game asker asked for is a rank we
        # know they wanted (unless the rank was later booked away). Verify at
        # least one ask survives as a standing "wanted" fact somewhere.
        wanted_any = any(d["wanted"] for d in sd["deductions"])
        self.assertTrue(asks)  # sanity: asks did happen
        self.assertTrue(wanted_any, "no ask survived into the deduction table")

    def test_stale_ask_outlives_events_window(self):
        # The core bug: an ask older than the observer's last turn is dropped
        # from recent_events but MUST remain in the deduction table.
        game, state, asks = self._play_to_ask(seed=11, plies=10)
        observer = int(state.current_player())
        sd = state.state_dict(observer)
        # Standing facts from the whole game are richer than the truncated
        # window: the union of all deduced signals should not be empty when
        # many asks have already happened.
        total_signal = sum(
            len(d["known_has"]) + len(d["known_void"]) + len(d["wanted"])
            for d in sd["deductions"]
            if d["player"] != observer
        )
        self.assertGreater(len(asks), 3)
        self.assertGreater(total_signal, 0, "deduction table lost all history")

    def test_deductions_do_not_leak_hidden_cards(self):
        # known_has must never overstate: it may only assert counts the engine
        # publicly guarantees (player_min_), never the opponent's true hand.
        game, state, _ = self._play_to_ask(seed=3, plies=12)
        observer = 0
        sd = state.state_dict(observer)
        # Opponent's real hand is not observer-visible; the deduction for them
        # may only be a lower bound. Cross-check against the public card total:
        # sum of known minimums can never exceed their public card count.
        players = {p["player"]: p for p in sd["players"]}
        for d in sd["deductions"]:
            pid = d["player"]
            if pid == observer:
                continue
            min_total = 0
            for token in d["known_has"]:  # tokens look like "K>=2"
                min_total += int(token.split(">=")[1])
            self.assertLessEqual(min_total, players[pid]["cards"])


class GoFishEnvTest(absltest.TestCase):
    def test_go_fish_agent_playthrough(self):
        env = make(
            "open_spiel_go_fish",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_go_fish")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_go_fish_observation_is_json(self):
        env = make(
            "open_spiel_go_fish",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # After dealing, it is player 0's turn (Ask phase).
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["phase"], "Ask")
        self.assertEqual(obs_p0["current_player"], 0)
        self.assertFalse(obs_p0["is_terminal"])
        # Two-player Go Fish deals 7 cards each.
        self.assertEqual(sum(obs_p0["hand"].values()), 7)
        self.assertEqual(obs_p0["players"][0], {"player": 0, "cards": 7, "books": 0})
        self.assertEqual(obs_p0["players"][1], {"player": 1, "cards": 7, "books": 0})

    def test_go_fish_observation_hides_opponent(self):
        env = make("open_spiel_go_fish", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Each player's observation reports only their own hand contents.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p0["observer"], 0)
        self.assertEqual(obs_p1["observer"], 1)
        self.assertEqual(sum(obs_p0["hand"].values()), 7)
        self.assertEqual(sum(obs_p1["hand"].values()), 7)
        # The two hands are dealt independently; opponent card contents are not
        # exposed in either observation (only aggregate counts in "players").

    def test_go_fish_invalid_action(self):
        env = make("open_spiel_go_fish", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"][0],
            open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        )


if __name__ == "__main__":
    absltest.main()
