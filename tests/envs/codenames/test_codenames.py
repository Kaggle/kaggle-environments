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
    
    # There should always be a winning team, so some reward must be > 0.
    # A team wins if they guess all words OR if the other team hits the assassin.
    assert sum(rewards) > 0

    print("Game successfully finished with rewards:", rewards)
    
if __name__ == "__main__":
    test_codenames_completes()
