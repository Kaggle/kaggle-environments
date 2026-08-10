import pytest

from kaggle_environments import make
from kaggle_environments.envs.word_association.word_association import (
    DEFAULT_INVALID_ACTION_REWARD,
)
from kaggle_environments.errors import DeadlineExceeded


def _legal_action(env, seat):
    """A clue for cluemasters, the first still-unrevealed index for guessers.

    Guessing a fixed index would forfeit the moment that square is revealed,
    which ends the whole episode -- so multi-game tests must pick a legal
    square each turn to actually reach a game transition.
    """
    if seat in (0, 2):
        return {"clue": "ANIMAL", "number": 1}
    revealed = env.state[0].observation.revealed
    return next((i for i, r in enumerate(revealed) if not r), -1)


def _step_legally(env):
    env.step([
        None if env.state[i].status != "ACTIVE" else _legal_action(env, i)
        for i in range(4)
    ])


def _assert_forfeited(state, offending_seat):
    """An illegal action ends the episode as a team-aware forfeit: every seat
    goes DONE, the offender's team takes DEFAULT_INVALID_ACTION_REWARD and the
    opposing team its negation. Mirrors open_spiel_env's non-strict INVALID
    path, except the loss is scoped to the team rather than the lone seat --
    crediting the offender's partner would reward a team for its own foul.
    """
    assert [s.status for s in state] == ["DONE"] * 4
    offending_team = [0, 1] if offending_seat in (0, 1) else [2, 3]
    for i, s in enumerate(state):
        expected = (
            DEFAULT_INVALID_ACTION_REWARD
            if i in offending_team
            else -DEFAULT_INVALID_ACTION_REWARD
        )
        assert s.reward == expected, f"seat {i}: expected {expected}, got {s.reward}"


def test_word_association_completes():
    env = make("word_association")
    
    # Run a full game using the random agent on all 4 slots.
    # The random agent will pass random clues and guesses.
    env.run(["random", "random", "random", "random"])
    
    # Assert that the game reaches a terminal state.
    assert env.done
    
    # Assert that 4 agents were present in the state list.
    assert len(env.state) == 4
    
    # Assert that the game ended properly and a winner was declared (rewards should be assigned)
    # Note: Kaggle environments automatically nullify rewards (None) for agents with INVALID status.
    rewards = [agent.reward if agent.reward is not None else -1 for agent in env.state]
    
    # Under the cumulative win logic, winning team gets 1 win, losing team gets 0 wins.
    # So the sum of rewards for the 4 agents should be 2 (two winning agents with 1.0).
    assert sum(rewards) == 2
    assert max(rewards) == 1
    assert min(rewards) == 0

    print("Game successfully finished with rewards:", rewards)

def test_random_start_counts():
    env = make("word_association")
    roles = env.state[0].observation.roles
    blue_count = sum(1 for r in roles if r == "blue")
    yellow_count = sum(1 for r in roles if r == "yellow")
    
    # One team must have 9, the other must have 8
    assert (blue_count == 9 and yellow_count == 8) or (blue_count == 8 and yellow_count == 9)
    
    # The starting team is determined by who has 9 words
    turn = env.state[0].observation.current_turn
    if blue_count == 9:
        assert turn == 0
    else:
        assert turn == 2

def test_minimum_one_guess():
    env = make("word_association")
    state = env.reset()
    turn = state[0].observation.current_turn
    
    env.step([{"clue": "VALID", "number": 2} if i == turn else None for i in range(4)])
    state = env.state
    guesser_turn = state[0].observation.current_turn
    
    # Try to pass immediately
    env.step([-1 if i == guesser_turn else None for i in range(4)])
    state = env.state
    
    _assert_forfeited(state, guesser_turn)
    assert env.done

def test_unlimited_clues_require_one_guess():
    env = make("word_association")
    state = env.reset()
    turn = state[0].observation.current_turn
    
    # Try with 0 clue
    env.step([{"clue": "ZERO", "number": 0} if i == turn else None for i in range(4)])
    state = env.state
    assert state[0].observation.guesses_remaining == 25
    
    guesser_turn = state[0].observation.current_turn
    env.step([-1 if i == guesser_turn else None for i in range(4)])
    _assert_forfeited(env.state, guesser_turn)

def test_infinity_clues_require_one_guess():
    env = make("word_association")
    state = env.reset()
    turn = state[0].observation.current_turn
    
    # Try with -1 (infinity) clue
    env.step([{"clue": "UNLIMITED", "number": -1} if i == turn else None for i in range(4)])
    state = env.state
    assert state[0].observation.guesses_remaining == 25
    
    guesser_turn = state[0].observation.current_turn
    env.step([-1 if i == guesser_turn else None for i in range(4)])
    _assert_forfeited(env.state, guesser_turn)

def test_clue_validation():
    env = make("word_association")
    state = env.reset()
    turn = state[0].observation.current_turn
    
    words = state[0].observation.words
    first_word = words[0]
    opponent_team = "yellow" if turn == 0 else "blue"
    opp_before = sum(1 for i in range(25) if state[0].observation.roles[i] == opponent_team and not state[0].observation.revealed[i])
    
    env.step([{"clue": first_word[1:4], "number": 1} if i == turn else None for i in range(4)])
    state = env.state
    
    opp_after = sum(1 for i in range(25) if state[0].observation.roles[i] == opponent_team and not state[0].observation.revealed[i])
    assert opp_after == opp_before - 1
    assert state[0].observation.current_turn == (2 if turn == 0 else 0)

def test_space_hyphen_validation():
    env = make("word_association")
    state = env.reset()
    turn = state[0].observation.current_turn
    
    opponent_team = "yellow" if turn == 0 else "blue"
    opp_before = sum(1 for i in range(25) if state[0].observation.roles[i] == opponent_team and not state[0].observation.revealed[i])
    
    # Try clue with space
    env.step([{"clue": "TWO WORDS", "number": 1} if i == turn else None for i in range(4)])
    state = env.state
    
    opp_after = sum(1 for i in range(25) if state[0].observation.roles[i] == opponent_team and not state[0].observation.revealed[i])
    assert opp_after == opp_before - 1
    assert state[0].observation.current_turn == (2 if turn == 0 else 0)
    
    # Reset for hyphen test
    state = env.reset()
    turn = state[0].observation.current_turn
    opponent_team = "yellow" if turn == 0 else "blue"
    opp_before = sum(1 for i in range(25) if state[0].observation.roles[i] == opponent_team and not state[0].observation.revealed[i])
    
    # Try clue with hyphen
    env.step([{"clue": "HYPHEN-ATED", "number": 1} if i == turn else None for i in range(4)])
    state = env.state
    
    opp_after = sum(1 for i in range(25) if state[0].observation.roles[i] == opponent_team and not state[0].observation.revealed[i])
    assert opp_after == opp_before - 1
    assert state[0].observation.current_turn == (2 if turn == 0 else 0)


def test_multi_game_cumulative_rewards():
    # Configure environment with 3 games per episode
    env = make("word_association", configuration={"games_per_episode": 3})
    
    # Run a full episode of multiple games using random agents
    env.run(["random", "random", "random", "random"])
    
    # Assert that the episode successfully completed
    assert env.done
    
    # Assert that 4 agents were present in the state list
    assert len(env.state) == 4
    
    # Retrieve the blue and yellow wins from the last state's observation
    obs = env.state[0].observation
    blue_wins = obs.blue_wins
    yellow_wins = obs.yellow_wins
    
    # Retrieve cumulative rewards for all agents
    rewards = [agent.reward if agent.reward is not None else 0 for agent in env.state]
    
    # For Blue team (agents 0 and 1), their rewards should equal blue_wins
    assert rewards[0] == blue_wins
    assert rewards[1] == blue_wins
    
    # For Yellow team (agents 2 and 3), their rewards should equal yellow_wins
    assert rewards[2] == yellow_wins
    assert rewards[3] == yellow_wins
    
    # The sum of wins should equal games_per_episode (3)
    assert blue_wins + yellow_wins == 3
    # Sum of all 4 agent rewards should be 2 * (blue_wins + yellow_wins) = 6
    assert sum(rewards) == 6

def test_multi_game_memory_consistent_across_agents():
    # All 4 agents should see the same memory fields after multi-game runs;
    # the per-game reset must update every agent, not only state[0].
    env = make("word_association", configuration={"games_per_episode": 3, "seed": 7})
    env.run(["random", "random", "random", "random"])

    obs0 = env.state[0].observation
    for i in range(1, 4):
        oi = env.state[i].observation
        assert oi.current_game == obs0.current_game, f"agent {i} current_game mismatch"
        assert len(oi.history) == len(obs0.history), f"agent {i} history length mismatch"
        assert len(oi.current_game_turns) == len(obs0.current_game_turns), (
            f"agent {i} current_game_turns length mismatch"
        )
        assert oi.blue_wins == obs0.blue_wins
        assert oi.yellow_wins == obs0.yellow_wins


def test_first_game_prompt_has_no_multi_game_status():
    # On the first game (single-game session or game 0 of a multi-game
    # session) the status block should not render — there is nothing useful
    # to report and the single-game prompt should be unchanged.
    from kaggle_environments.envs.word_association.harness import generate_prompt
    for cfg in ({}, {"games_per_episode": 5, "seed": 0}):
        env = make("word_association", configuration=cfg)
        obs = env.state[0].observation
        prompt = generate_prompt(obs, [])
        assert "Current score" not in prompt, f"unexpected status block for cfg={cfg}"
        assert "most game wins overall" not in prompt, f"unexpected status block for cfg={cfg}"


def test_subsequent_game_prompt_has_status_block():
    # After the first game completes, the status line should appear with the
    # current game number and score — but never the total number of games.
    from kaggle_environments.envs.word_association.harness import generate_prompt
    env = make("word_association", configuration={"games_per_episode": 5, "seed": 0})
    env.reset()
    while env.state[0].observation.current_game == 0 and not env.done:
        _step_legally(env)
    assert env.state[0].observation.current_game >= 1

    obs = env.state[0].observation
    prompt = generate_prompt(obs, [])
    assert f"This is game {obs.current_game + 1}." in prompt
    assert "The team with the most game wins overall is the winner." in prompt
    assert f"Current score: BLUE {obs.blue_wins} – YELLOW {obs.yellow_wins}." in prompt
    # Total games count must not be leaked into the prompt.
    assert "of 5" not in prompt
    assert "5 games" not in prompt


def test_multi_game_guessers_dont_see_unmasked_roles_at_transition():
    # After each per-episode game transition, guessers (agents 1 and 3) must
    # see roles masked as "Unknown" — initialize_game writes full roles to
    # every agent, so update_visibility must run before the snapshot is returned.
    env = make("word_association", configuration={"games_per_episode": 5, "seed": 0})
    env.reset()
    prev_cg = env.state[0].observation.current_game
    saw_transition = False
    while not env.done:
        _step_legally(env)
        cg = env.state[0].observation.current_game
        if cg != prev_cg:
            saw_transition = True
            for guesser in (1, 3):
                roles = env.state[guesser].observation.roles
                non_unknown = sum(1 for r in roles if r != "Unknown")
                # All squares start unrevealed, so guessers should see 0 real roles.
                assert non_unknown == 0, (
                    f"guesser {guesser} saw {non_unknown} unmasked roles "
                    f"at game-{cg} start"
                )
            prev_cg = cg
    assert saw_transition, "test never observed a game transition"


def test_multi_game_per_game_seed_uniqueness():
    # Different games within one episode must use different word boards.
    env = make("word_association", configuration={"games_per_episode": 2, "seed": 7})
    env.reset()
    words_game1 = list(env.state[0].observation.words)
    env.run(["random", "random", "random", "random"])
    words_game2 = list(env.state[0].observation.words)
    assert words_game1 != words_game2

    # First game must use the provided seed directly (matches a solo run).
    solo = make("word_association", configuration={"games_per_episode": 1, "seed": 7})
    solo.reset()
    assert list(solo.state[0].observation.words) == words_game1

    # Full episode is deterministic across runs.
    env2 = make("word_association", configuration={"games_per_episode": 2, "seed": 7})
    env2.run(["random", "random", "random", "random"])
    assert list(env2.state[0].observation.words) == words_game2


# --- Framework failure vs. illegal move -------------------------------------
#
# A seat that crashed or timed out is a broken participant, not a model making
# an illegal move, so the two must not collapse into the same outcome. Matching
# open_spiel_env's non-strict path: ERROR/TIMEOUT voids the episode (all seats
# ERROR, all rewards nulled), while an illegal action is a scored forfeit.


def _legal_agent(observation, configuration):
    if observation.current_turn in (0, 2):
        return {"clue": "ANIMAL", "number": 1}
    return next((i for i, r in enumerate(observation.revealed) if not r), -1)


@pytest.mark.parametrize("crash_seat", [0, 1, 2, 3])
def test_agent_crash_voids_the_episode(crash_seat):
    """A raising agent must not be laundered into an INVALID forfeit that
    credits the opposing team with a win it did not earn."""

    def crash(observation, configuration):
        raise RuntimeError("provider exploded")

    agents = [_legal_agent] * 4
    agents[crash_seat] = crash
    env = make("word_association", configuration={"seed": 3})
    env.run(agents)

    assert env.done
    assert [s.status for s in env.state] == ["ERROR"] * 4
    assert [s.reward for s in env.state] == [None] * 4


@pytest.mark.parametrize("timeout_seat", [0, 1, 2, 3])
def test_agent_timeout_voids_the_episode(timeout_seat):
    """A timed-out seat keeps TIMEOUT (which voids the episode the same way as
    ERROR) rather than being relabeled INVALID."""
    env = make("word_association", configuration={"seed": 3})
    env.reset()
    while True:
        turn = env.state[0].observation.current_turn
        actions = [None] * 4
        if turn == timeout_seat:
            actions[turn] = DeadlineExceeded()
            env.step(actions)
            break
        actions[turn] = _legal_agent(env.state[turn].observation, None)
        env.step(actions)

    assert env.done
    assert env.state[timeout_seat].status == "TIMEOUT"
    assert all(s.status in ("ERROR", "TIMEOUT") for s in env.state)
    assert [s.reward for s in env.state] == [None] * 4


def test_illegal_move_is_a_scored_forfeit_not_an_error():
    """The counterpart to the tests above: a well-formed but rule-breaking
    action still ends the episode with a real result."""

    def bad_guesser(observation, configuration):
        if observation.current_turn in (0, 2):
            return {"clue": "ANIMAL", "number": 1}
        return 999  # out of range

    env = make("word_association", configuration={"seed": 3})
    env.reset()
    # Whichever team leads, its guesser is the first seat to offend.
    offending_seat = env.state[0].observation.current_turn + 1
    env.run([_legal_agent, bad_guesser, _legal_agent, bad_guesser])

    assert env.done
    _assert_forfeited(env.state, offending_seat)


def test_forfeit_ends_a_multi_game_episode_immediately():
    """A forfeit is terminal: the next game must not start and relaunder the
    forfeiting seats back to ACTIVE."""

    def bad_guesser(observation, configuration):
        if observation.current_turn in (0, 2):
            return {"clue": "ANIMAL", "number": 1}
        return 999

    env = make(
        "word_association",
        configuration={"games_per_episode": 5, "seed": 0},
    )
    # Whichever team leads, its guesser is the first seat to submit the
    # out-of-range guess.
    env.reset()
    offending_seat = env.state[0].observation.current_turn + 1
    env.run([_legal_agent, bad_guesser, _legal_agent, bad_guesser])

    assert env.done
    assert env.state[0].observation.current_game == 0
    _assert_forfeited(env.state, offending_seat)


if __name__ == "__main__":
    test_word_association_completes()
    test_random_start_counts()
    test_minimum_one_guess()
    test_unlimited_clues_require_one_guess()
    test_infinity_clues_require_one_guess()
    test_clue_validation()
    test_space_hyphen_validation()
    test_multi_game_cumulative_rewards()
    test_multi_game_memory_consistent_across_agents()
    test_first_game_prompt_has_no_multi_game_status()
    test_subsequent_game_prompt_has_status_block()
    test_multi_game_guessers_dont_see_unmasked_roles_at_transition()
    test_multi_game_per_game_seed_uniqueness()
    print("All Word Association rule tests passed!")