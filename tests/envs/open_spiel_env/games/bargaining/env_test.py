"""Env-level tests for open_spiel_bargaining."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env

_AGREE = 120


def _obs(env, player: int) -> dict:
    return json.loads(env.state[player]["observation"]["observationString"])


class BargainingEnvTest(absltest.TestCase):
    def test_bargaining_agent_playthrough(self):
        env = make(
            "open_spiel_bargaining",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_bargaining")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_bargaining_observation_is_json_and_hides_opponent_values(self):
        env = make("open_spiel_bargaining", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Chance / setup step.
        obs_p0 = _obs(env, 0)
        obs_p1 = _obs(env, 1)
        self.assertEqual(obs_p0["current_player"], 0)
        self.assertEqual(obs_p0["viewing_player"], 0)
        self.assertEqual(obs_p1["viewing_player"], 1)
        # Pool is public; both players see the same item counts.
        self.assertEqual(obs_p0["pool"], obs_p1["pool"])
        self.assertEqual(set(obs_p0["pool"].keys()), {"book", "hat", "basketball"})
        # Private valuations differ (with overwhelming probability for the
        # default chance distribution) and the other player's are not exposed.
        self.assertEqual(set(obs_p0["my_values"].keys()), {"book", "hat", "basketball"})
        self.assertNotIn("opponent_values", obs_p0)
        # No moves yet: empty offer history, no agreement, no returns.
        self.assertFalse(obs_p0["is_terminal"])
        self.assertFalse(obs_p0["agreement_reached"])
        self.assertEqual(obs_p0["num_offers"], 0)
        self.assertEqual(obs_p0["offer_history"], [])
        self.assertIsNone(obs_p0["last_offer"])
        self.assertIsNone(obs_p0["returns"])
        self.assertEqual(obs_p0["params"]["agree_action"], _AGREE)
        self.assertEqual(obs_p0["params"]["max_turns"], 10)

    def test_bargaining_acceptance_terminates_and_assigns_returns(self):
        env = make("open_spiel_bargaining", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Chance step.
        # Player 0 proposes keeping everything in the pool for themselves
        # (action 0 = "Offer: Book: 0, Hat: 0, Basketball: 0" -- P0 keeps 0 of
        # each, opponent gets the entire pool). Player 1 then accepts.
        obs_p0_initial = _obs(env, 0)
        pool = obs_p0_initial["pool"]
        p1_values = _obs(env, 1)["my_values"]
        # Pick a non-Agree legal offer for P0: action 0 always works since 0
        # items <= pool count for every item.
        env.step([{"submission": 0}, {"submission": -1}])
        # P1 agrees.
        env.step([{"submission": -1}, {"submission": _AGREE}])
        self.assertTrue(env.done)
        final_p0 = _obs(env, 0)
        final_p1 = _obs(env, 1)
        self.assertTrue(final_p0["is_terminal"])
        self.assertTrue(final_p0["agreement_reached"])
        # The accept action shows up as an "agree" event in offer_history.
        self.assertEqual(final_p0["offer_history"][-1], {"player": 1, "type": "agree"})
        self.assertEqual(
            final_p0["offer_history"][0],
            {
                "player": 0,
                "type": "offer",
                "items": {"book": 0, "hat": 0, "basketball": 0},
            },
        )
        # Both players' terminal returns are exposed.
        self.assertIsNotNone(final_p0["returns"])
        self.assertEqual(final_p0["returns"], final_p1["returns"])
        # P0 kept nothing, so their utility is 0. P1 got the whole pool, so
        # their utility is the dot product of their values with the pool.
        rewards = env.toJSON()["rewards"]
        self.assertEqual(rewards[0], 0.0)
        expected_p1 = sum(p1_values[item] * pool[item] for item in pool)
        self.assertEqual(rewards[1], expected_p1)

    def test_bargaining_discount_param_propagates_and_discounts_returns(self):
        # discount=0.5 must (a) reach the underlying OpenSpiel game via the
        # generic openSpielGameParameters plumbing, (b) surface in
        # state.params, and (c) actually scale terminal returns. We force a
        # 3-offer sequence so the cumulative discount is 0.5^(3-1) = 0.25 by
        # OpenSpiel's rule (discount applied on move_number_ >= 3, i.e.
        # starting from P0's 2nd action).
        env = make(
            "open_spiel_bargaining",
            configuration={"openSpielGameParameters": {"discount": 0.5}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Chance step.
        obs_p0_initial = _obs(env, 0)
        self.assertEqual(obs_p0_initial["params"]["discount"], 0.5)
        pool = obs_p0_initial["pool"]
        p1_values = _obs(env, 1)["my_values"]

        # P0 offer 1: keep nothing.
        env.step([{"submission": 0}, {"submission": -1}])
        # P1 offer 2: also keep nothing (so opponent's-perspective offers stay symmetric).
        env.step([{"submission": -1}, {"submission": 0}])
        # P0 accepts P1's offer (3rd action by a player after chance) -> N=2 offers
        # before agreement, discount factor 0.5^(2-1) = 0.5 applied.
        env.step([{"submission": _AGREE}, {"submission": -1}])

        self.assertTrue(env.done)
        rewards = env.toJSON()["rewards"]
        # P0 accepted P1's "keep nothing" offer -> P0 gets the whole pool.
        # Undiscounted P0 reward = sum(p0_values[i] * pool[i]).
        p0_values = _obs(env, 0)["my_values"]
        undiscounted_p0 = sum(p0_values[item] * pool[item] for item in pool)
        self.assertEqual(rewards[0], 0.5 * undiscounted_p0)
        # P1 gets nothing (their offer kept nothing for themselves), so 0
        # regardless of discount.
        del p1_values  # only used for symmetry comment above
        self.assertEqual(rewards[1], 0.0)

    def test_bargaining_invalid_action(self):
        env = make("open_spiel_bargaining", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Chance step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )


if __name__ == "__main__":
    absltest.main()
