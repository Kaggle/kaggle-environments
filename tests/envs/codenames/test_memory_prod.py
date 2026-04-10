from kaggle_environments import make

def test_codenames_memory_initialization():
    env = make("codenames", configuration={"games_per_episode": 2})
    state = env.reset()
    assert "current_game" in state[0].observation
    assert state[0].observation.current_game == 0
    assert "history" in state[0].observation
    assert len(state[0].observation.history) == 0

def test_codenames_memory_completes_multiple_games():
    env = make("codenames", configuration={"games_per_episode": 2})
    
    # Run a full game using random agents
    env.run(["random", "random", "random", "random"])
    
    # After running, we expect it to be DONE after 2 games
    assert env.done
    
    # Check history
    obs = env.state[0].observation
    assert "history" in obs
    assert len(obs.history) == 2
    assert obs.history[0]["game"] == 0
    assert obs.history[1]["game"] == 1
    
    print("Production memory game successfully finished with history length:", len(obs.history))

def test_codenames_memory_window_size():
    # Run 3 games but set window size to 1
    env = make("codenames", configuration={"games_per_episode": 3, "memory_window_size": 1})
    
    env.run(["random", "random", "random", "random"])
    
    assert env.done
    
    obs = env.state[0].observation
    assert "history" in obs
    # History should only contain the LAST game (game index 2)
    assert len(obs.history) == 1
    assert obs.history[0]["game"] == 2
    
    print("Memory window size test passed!")

if __name__ == "__main__":
    test_codenames_memory_initialization()
    test_codenames_memory_completes_multiple_games()
    test_codenames_memory_window_size()
    print("All production memory tests passed!")
