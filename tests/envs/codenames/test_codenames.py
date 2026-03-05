from kaggle_environments import make

def test_codenames_completes():
    env = make("codenames")
    
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
    
    # Under the new logic, winning team gets +1, losing team gets -1. 
    # The sum of all rewards should be 0, and max should be 1.
    assert sum(rewards) == 0
    assert max(rewards) == 1
    assert min(rewards) == -1

    print("Game successfully finished with rewards:", rewards)
    
if __name__ == "__main__":
    test_codenames_completes()

def test_random_start_counts():
    env = make("codenames")
    roles = env.state[0].observation.roles
    red_count = sum(1 for r in roles if r == "red")
    blue_count = sum(1 for r in roles if r == "blue")
    
    # One team must have 9, the other must have 8
    assert (red_count == 9 and blue_count == 8) or (red_count == 8 and blue_count == 9)
    
    # The starting team is determined by who has 9 words
    turn = env.state[0].observation.current_turn
    if red_count == 9:
        assert turn == 0
    else:
        assert turn == 2

def test_minimum_one_guess():
    env = make("codenames")
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

def test_unlimited_clues():
    env = make("codenames")
    state = env.reset()
    turn = state[0].observation.current_turn
    
    env.step([{"clue": "UNLIMITED", "number": 0} if i == turn else None for i in range(4)])
    state = env.state
    
    assert state[0].observation.guesses_remaining == 25

def test_clue_validation():
    env = make("codenames")
    state = env.reset()
    turn = state[0].observation.current_turn
    
    words = state[0].observation.words
    first_word = words[0]
    opponent_team = "blue" if turn == 0 else "red"
    opp_before = sum(1 for i in range(25) if state[0].observation.roles[i] == opponent_team and not state[0].observation.revealed[i])
    
    env.step([{"clue": first_word[1:4], "number": 1} if i == turn else None for i in range(4)])
    state = env.state
    
    opp_after = sum(1 for i in range(25) if state[0].observation.roles[i] == opponent_team and not state[0].observation.revealed[i])
    assert opp_after == opp_before - 1
    assert state[0].observation.current_turn == (2 if turn == 0 else 0)

test_random_start_counts()
test_minimum_one_guess()
test_unlimited_clues()
test_clue_validation()
print("All Codenames rule tests passed!")
