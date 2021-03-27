from kaggle_environments import make
from .agents import rock, scissors
if __name__ == "__main__":
    print("init")
    env = make("luxai2021", configuration={"episodeSteps": 4}, debug=True)
    # env.run([rock, scissors])
    # print(env.toJSON())
    trainer = env.train([None, scissors])
    for i in range(2):
        # env.render()
        action = 0
        print("=== Episode {} === ".format(i + 1))
        obs, reward, done, info = trainer.step(action)
        if done:
            obs = trainer.reset()