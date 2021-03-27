from kaggle_environments import make
from .agents import rock, scissors
if __name__ == "__main__":
    print("init")
    env = make("rps", configuration={"episodeSteps": 4}, debug=True)
    # env.run([rock, scissors])
    # print(env.toJSON())
    trainer = env.train([None, scissors])
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