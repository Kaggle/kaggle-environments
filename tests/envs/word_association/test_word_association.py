from kaggle_environments import make

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
    
    assert state[guesser_turn].status == "INVALID"
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
    assert env.state[guesser_turn].status == "INVALID"

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
    assert env.state[guesser_turn].status == "INVALID"

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

if __name__ == "__main__":
    test_word_association_completes()
    test_random_start_counts()
    test_minimum_one_guess()
    test_unlimited_clues_require_one_guess()
    test_infinity_clues_require_one_guess()
    test_clue_validation()
    test_space_hyphen_validation()
    test_multi_game_cumulative_rewards()
    print("All Word Association rule tests passed!")