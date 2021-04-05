from kaggle_environments import make
from kit.game import Game
if __name__ == "__main__":
    print("init")
    env = make("lux_ai_2021", configuration={"episodeSteps": 201, "saveReplays": False}, debug=True)
    # env.run([rock, statistical])
    # print(env.toJSON())

    trainer = env.train([None, "random_agent"])
    eps = 1
    # game_state = Game()
    # game_state._initialize(env.state[0].observation["updates"])
    
    for i in range(200):
        
        # env.render()
        action = 0
        print("=== Episode {} - Step {} === ".format(eps, i + 1))
        obs, reward, done, info = trainer.step(action)
        print({"reward": reward, "obs_reward": obs["reward"]})
        # print(repr(game_state.game_map.map[0][0]))
        # update the game state for next step
        # game_state._update(obs["updates"])
        if done:
            # if episode is done, reset env and get new observations and new game state
            obs = trainer.reset()
            # game_state = Game()
            # game_state._initialize(obs["updates"])
            eps += 1