# Kore Fleets typescript bot

## Requirements

1. node and npm/npx/yarn
2. typescript installed
3. python3 installed
4. kaggle_environments pip package installed

## Getting started

1. `npm install`

## Running Locally

1. transpile your bot with `npm run compile` or `tsc`
2. run a match with `npm watch4`

* if you don't have google-chrome installed, open the replay.html in the browser of your choice
* see package.json for more options

## Creating a submission

1. `npm run package`
2. upload submission.tar.gz to kaggle
3. profit!

## Running tests

1. `npm test`

## Interpreter and training

A basic TS interpreter has been created in `interpreter.ts`. You can use or modify this file to train machine learning models in JS/TS.

Currently it supports 2 agents and customizable number of episodes. 

It has two modes: `run` and `step`.

`run` mode: After each episode, you can access the complete history of the game. For each turn, you can access the full observation (state) as a Board object, actions performed and the reward obtained after performing the action. This mode is useful for evaluating an agent.

`step` mode: The interpreter initializes new games and allows stepping through the game interactively. You have complete control over the board and the agent during each step. This mode is useful for training machine learning models.

Sample command to run the interpreter can be found in npm scripts as `npm run interpreter:run` and `npm run interpreter:step`.

## Miner bot and Do nothing bot

A sample miner bot `MinerBot.ts` is provided, with Python entrypoint as `miner.py`. It has the same logic as the Python `miner` bot in `kore_fleets.py`.

To run it aginst Python miner bot with TS interpreter for 20 episodes:

1. `npm run compile`
2. `node --require ts-node/register interpreter.ts 20 ./miner.py miner`

A sample do nothing bot `DoNothingBot.ts` is also provided.
