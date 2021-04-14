# Dimensions

[![npm version](https://badge.fury.io/js/dimensions-ai.svg)](https://badge.fury.io/js/dimensions-ai)

This is an **open sourced** **generic** **Artificial Intelligence competition framework**, intended to provide you the fully scalable infrastructure needed to run your own AI competition with no hassle.

All you need to do?

Code a **competition design** and code a **bot**

Dimensions handles the rest, including match and tournament running, security, scalability and a local API and website through which you can monitor and control the entire system.

The framework was built with the goals of being **generalizable** and **accessible**. That's why Dimensions utilizes an I/O based model to run competitions and pit AI agents against each other (or themselves!), allowing it to be generic and language agnostic so anyone from any background can compete in your competition design.

It generalizes to many scenarios, and is able to recreate a range of systems from the [Halite 3 AI competition](https://halite.io/), [Battlecode 2020](https://battlecode.org), to a generalized [Open AI Gym](https://gym.openai.com/) that is open to machine learning in all languages in addition to Python through simple stdio.

This was inspired by [Battlecode](battlecode.org/) and [Halite](https://halite.io/)

Keep reading to learn how to [get started](#getting-started) and make a tournament like this:

![dimensions-trueskill-RPS](assets/dimensions-trueskill-RPS.gif)

Of which the [AI bots](https://github.com/StoneT2000/Dimensions/blob/master/tests/kits/js/normal/rock.js) are all coded in about 10 lines or less (ignoring the starter kit)

```js
const kit = require('./kit');
const agent = new kit.Agent();
agent.initialize().then(async () => {
  while (true) {
    console.log('R'); // tell the match you want to play Rock in the game
    agent.endTurn(); // end turn
    await agent.update(); // wait for updates
  }
});
```

As another proof of concept of how seamless and generalizable this framework is, see [the recreation of Halite 3](https://github.com/StoneT2000/dimensions-halite3) using this framework.

Follow these links to jump straight to [Documentation](https://stonet2000.github.io/Dimensions/index.html), [Contributing](#contributing), [Development](#development) or [Plans](#plans) curated by the owner and the community.

Also checkout the blog post introducing the motivation for Dimensions and thoughts about it here: https://stonet2000.github.io/blog/posts/Dimensions/index.html

## Features

- Easy to build an AI competition that is language agnostic, allowing any kind of bot in any language to compete in your competition
- Run many kinds of AI competitions and run different kinds of formats like round robin or using Trueskill in a ladder tournament (like a leaderboard).
- Wrap your own AI competition built without the dimensions framework to make use of its competition running features such as Trueskill ranking like the [Halite 4 wrapper](https://github.com/StoneT2000/Halite-4-Tournament-Runner) used by competitors in the Halite 4 challenge on Kaggle.
- Comes with an API served locally that gives access to data on ongoing matches and tournaments and allows for direct control of matches and tournaments through the API. See this page for details on this API: https://github.com/StoneT2000/Dimensions/wiki/Dimensions-Station-API
- Supports plugins like the [MongoDB](https://github.com/StoneT2000/Dimensions/wiki/Plugin#supported-plugins) plugin that takes three lines of code to automatically integrate and scale up your tournament and integrate an automatic user authentication and login system. See [this](https://github.com/StoneT2000/Dimensions/wiki/Scaling) for complete info on how to scale up.
- Ensures malicious bots cannot cause harm to your servers through `secureMode`. See [this wiki page](https://github.com/StoneT2000/Dimensions/wiki/Security) for details.
- Built with Typescript, meaning flexibility in coding any design, and easily integrates into a frontend replay viewer to reconstruct matches for viewing using minimal replay data storage.

## Requirements

At the moment, MacOS and Linux are 100% supported. Windows platforms might not work and it is highly suggested to install [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10). It's also suggested to use Node 12.x or above. Lower versions are untested.

## Getting Started

This guide will take you through how to start and run a competition built with Javascript/Typescript. To see how to use this framework to run an AI competition built without the dimensions framework, see [this wiki page](https://github.com/StoneT2000/Dimensions/wiki/Custom-Competition-Design).

First, install the `dimensions-ai` package

```
npm install dimensions-ai
```

Create a new file called `run.js` and inside it we need to first `require` the package

```js
const Dimension = require('dimensions-ai');
```

In order to start writing AI to compete against each other in a competition, you need to do two things.

1. Design the competition
2. Design an AI starter kit

You need to design a competition to allow people to compete and facilitate the matches. More info on that soon. it is highly suggested to design an AI starter kit so people can get straight into competing.

If you already have a design, feel free to skip to the section on running a [match](#run-a-match) and a [tournament](#run-a-tournament)

### Designing The Competition

It is suggested to design the competition in Javascript / Typescript using the Dimensions framework. See https://github.com/StoneT2000/Dimensions/wiki/Creating-a-Design for a tutorial on creating a design.

You can also create a custom design outside of the framework, see https://github.com/StoneT2000/Dimensions/wiki/Custom-Competition-Design for a tutorial on that.

### Designing an AI Starter Kit

Example starter kits can be found in the [/templates/starter-kits](https://github.com/StoneT2000/Dimensions/tree/master/templates/starter-kits/) folder and you can just copy these and tweak them accordingly to your own design.

See https://github.com/StoneT2000/Dimensions/wiki/Creating-a-Starter-kit for a tutorial on creating a starter kit.

### Run a Match

Now with a design done and a starter kit created, all you have to do is write a quick AI that does something and then run a match as follows:

First initialize your design and pass it a name. Then create a new `dimension` with `Dimension.create`.

```js
let RPSDesign = new RockPaperScissorsDesign('RPS!');
let myDimension = Dimension.create(RPSDesign);
```

We can now run our first match by passing in an array of paths to the bot codes, each of which will generate into a new agent that participates in the match. You can then also pass in any configurations you want accessible through `match.configs` in the life cycle functions of your `design`.

```js
let results = await myDimension.runMatch(
  [
    './examples/rock-paper-scissors/bots/paper.js',
    './examples/rock-paper-scissors/bots/rock.js',
  ],
  {
    bestOf: 5, // a configuration accessible in match through match.configs.bestOf
  }
);
```

You can now log the results, of which are the same results returned by your `design's` `getResult` function.

```js
console.log(results);
```

Notice that your console will also print something about a station. It'll give you a link to the `Station`, a local server that gives you access to an API to access and control your Dimension, Matches, Tournaments and more. Check out https://github.com/StoneT2000/Dimensions/wiki/Dimensions-Station-API for details on the API.

If you want to view the API from a website, see this repo: https://github.com/StoneT2000/Dimensions-web

### Run a Tournament

This framework also provides tournament running features, which currently include [Elimination](https://stonet2000.github.io/Dimensions/classes/_tournament_elimination_index_.elimination.html), and a flexible [Ladder](https://stonet2000.github.io/Dimensions/classes/_tournament_ladder_index_.ladder.html) type tournaments. Additionally, there are various ranking systems used, such as Win/Tie/Loss and Trueskill. This section takes your through a really brief rundown of how to run a tournament. See [this wiki page](https://github.com/StoneT2000/Dimensions/wiki/Running-Tournaments) for more in depth details on setting up the various kinds of tournaments

Here is how you run a tournament. First, you will need a `resultHandler` function. This function must be given to the tournament to indicate how the results of a `match` should be interpreted. Recall that these results are returned by the `getResult` function in your design class. It is suggested to provide these result handlers in your `Design`.

Next, you need to pass in some required configurations, namely `type, rankSystem, agentsPerMatch, resultHandler`. The following code snippet shows an example.

```js
let RPSDesign = new RockPaperScissorsDesign('RPS!');
let myDimension = Dimension.create(RPSDesign);
let Tournament = Dimension.Tournament;
let simpleBot = './bots/rock.js';
let botSources = [simpleBot, simpleBot, simpleBot, simpleBot, simpleBot];

let RPSTournament = myDimension.createTournament(botSources, {
  name: 'A Best of 329 Rock Paper Scissors Tournament', // give it a name
  type: Tournament.Type.LADDER, // Create a Ladder Tournament
  rankSystem: Tournament.RankSystem.TRUESKILL, // Use Trueskill to rank bots
  agentsPerMatch: [2], // specify how many bots can play at a time
  defaultMatchConfigs: {
    bestOf: 329,
    loggingLevel: Dimension.Logger.Level.NONE, // turn off match logging
  },
  resultHandler: (results) => {
    let ranks = [];
    if (results.winner === 'Tie') {
      ranks = [
        { rank: 1, agentID: 0 },
        { rank: 1, agentID: 1 },
      ];
    } else {
      let loserID = (results.winnerID + 1) % 2;
      ranks = [
        { rank: 1, agentID: results.winnerID },
        { rank: 2, agentID: loserID },
      ];
    }
    return {
      ranks: ranks,
    };
  },
});

RPSTournament.run();
```

Once running, the console will display a live leaderboard of the tournament running, showing the current ranks and scores of all the bots, similar to the gif shown at the beginning of this document.

Documentation / guides on the tournaments and how to use them can be found https://github.com/StoneT2000/Dimensions/wiki/Running-Tournaments. Full documentation on Tournaments can be found [here](https://stonet2000.github.io/Dimensions/interfaces/_tournament_index_.tournament.tournamentconfigsbase.html).

Note that different tournament types have different tournament configurations and different rank systems have different ranking configurations, all of which can be found on the documentation.

### More Stuff!

The [wiki](https://github.com/StoneT2000/Dimensions/wiki) is populated with more basic and advanced example usages of this framework. This ranges from how to [configure the match engine](https://github.com/StoneT2000/Dimensions/wiki/Configuration#engine-options), [configuring various tournaments and rank systems](https://github.com/StoneT2000/Dimensions/wiki/Running-Tournaments), to tips on designing a successful competition.

### Strong Recommendations

In a production setting, it is strongly recommended to create a Dimension in `secureMode` to decrease the likelihood of user uploaded bot code of causing any significant harm to a server. By default, `secureMode` is set to false, but you will always get a warning about it. Setting it to true only requires you to install [Docker](https://www.docker.com/get-started).

## Plugins

Plugins intend to be a simple "drag and drop." Dimensions can `use` a plugin and the plugin will automatically configure the dimension as needed. See here for more [info on the available plugins](https://github.com/StoneT2000/Dimensions/wiki/Plugin#supported-plugins). See here for how to [develop a plugin](https://github.com/StoneT2000/Dimensions/wiki/Plugin#developing-a-plugin)

For example, here's two lines of code that integrate MongoDB as a database:

```js
let mongo = new Dimension.MongoDB('mongodb://localhost:27017/dimensions');
await myDimension.use(mongo);
```

## Contributing

Everyone is more than welcome to contribute to this project! You can open an issue or submit a PR

Check out the issues for this repository to get an idea on something you can help out with!

## Development

This is all written in [TypeScript](https://www.typescriptlang.org/)

First install all necessary packages and pull necessary docker images for testing with

```
npm install
./pulla_test_docker_images.sh
```

Start development by running

```
npm run watch
```

to watch for code changes in the `src` folder and reload the build folder. Note this does not build any frontend code.

Tests are built with [Mocha](https://mochajs.org/) and [Chai](https://www.chaijs.com/). You will need mongodb setup serving through port `27017` to run database plugin tests.

If that is setup, run tests with

```
npm run test
```

Run

```
npm run build
```

to build the entire library.

Run

```
npm run docs
```

to generate documentation

## Plans

- Make it easier to create a `Design` (design a competition)
  - Make README easier to READ, and reduce the initial "getting-used-to-framework" curve.
- Make it easier for users to dive deeper into the `MatchEngine`, `Matches`, `Dimensions` to give them greater flexibility over the backend infrastructure
  - At the moment, there are plans for a parallel command stream option, where all agents send commands whenever they want and the engine just sends them to the update function
  - Allow users to tinker the MatchEngine to their needs somehow. (Extend it as a class and pass it to Dimensions)
- Security Designs to help ensure that users won't create `Designs` susceptible to cheating and match breaking behavior from bots participating in a `Match`
  - Give some guidelines
  - Add some options and default values for certain configurations, e.g.
    - Max command limit per `timeStep` (for a game of rock paper scissors, this would be 1, it wouldn't make sense to flood the `MatchEngine` with several commands, which could break the `Match`)
- Add visualizers for rock paper scissors example and domination example (and others if possible)
- Generalize a match visualizer
- Add more example `Designs` and starter kits for other popular ai games
  - Recreate Kaggle Simulation's xConnect
