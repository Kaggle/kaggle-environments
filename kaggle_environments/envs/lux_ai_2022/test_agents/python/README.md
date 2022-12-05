# Lux AI Season 2 Python Kit

This is the folder for the Python kit. Please make sure to read the instructions as they are important regarding how you will write a bot and submit it to the competition.

Make sure to check our [Discord](https://discord.gg/aWJt3UAcgn) or the [Kaggle forums](https://www.kaggle.com/c/lux-ai-2022-beta/discussion) for announcements if there are any breaking changes.

## Requirements

You will need Python 3.7 or higher and NumPy installed (which should come with the dependencies you installed for the environment)

## Getting Started

To get started, download this folder from this repository.

Your core agent code will go into `agent.py`, and you can create and use more files to help you as well. You should leave `main.py` alone as that code enables your agent to compete against other agents locally and on Kaggle.

To quickly test run your agent, run

```
luxai2022 main.py main.py --out=replay.json
```

This will run the `agent.py` code in the same folder as `main.py` and generate a replay file saved to `replay.json`.

## Developing
Now that you have the code up and running, you are ready to start programming and having some fun!

If you haven't read it already, take a look at the [design specifications for the competition](https://www.lux-ai.org/specs-2022-beta). This will go through the rules and objectives of the competition. For a in-depth tutorial, we provide a jupyter notebook both [locally](https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/kits/python/starter_python.ipynb) and on [Kaggle](https://www.kaggle.com/code/stonet2000/lux-ai-season-2-jupyter-notebook-tutorial#Lux-AI-Season-2-Jupyter-Notebook-Tutorial---Python-Kit)

All of our kits follow a common API through which you can use to access various functions and properties that will help you develop your strategy and bot. The markdown version is here: https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/kits/README.md, which also describes the observation and action structure/spaces.

## Submitting to Kaggle

Submissions need to be a .tar.gz bundle with main.py at the top level directory (not nested). To create a submission, create the .tar.gz with `tar -czvf submission.tar.gz *`. Upload this under the [My Submissions tab] and you should be good to go! Your submission will start with a scheduled game vs itself to ensure everything is working before being entered into the matchmaking pool against the rest of the leaderboard.

## FAQ

As questions come up, this will be populated with frequently asked questions regarding the Python kit.