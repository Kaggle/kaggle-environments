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
    rewards = [agent.reward for agent in env.state]
    
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

test_random_start_counts()
