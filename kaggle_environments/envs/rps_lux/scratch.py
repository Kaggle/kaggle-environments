from kaggle_environments import make
from .agents import rock, scissors, statistical
if __name__ == "__main__":
    print("init")
    env = make("rps_lux", configuration={"episodeSteps": 4}, debug=True)
    # env.run([rock, statistical])
    # print(env.toJSON())

    trainer = env.train([None, statistical])
    eps = 1
    for i in range(10):
        # env.render()
        action = 0
        print("=== Episode {} - Step {} === ".format(eps, i + 1))
        obs, reward, done, info = trainer.step(action)
        print(obs)
        if done:
            obs = trainer.reset()
            eps += 1