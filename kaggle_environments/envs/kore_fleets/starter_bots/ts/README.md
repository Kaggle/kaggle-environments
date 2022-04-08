# Kore Fleets typescript bot

## Requirements

1. node and npm/npx/yarn
2. typescript installed

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

Currently it supports 2 agents and customizable number of episodes. After each episode, you can access the complete history of the game. For each turn, you can access the full observation (state) as a Board object, actions performed and the reward obtained after performing the action.

Sample command to run the interpreter can be found in npm scripts as `npm run interpreter`.
