const kit = require('./lux/kit');
const GAME_CONSTANTS = require('./lux/game_constants');
const DIRECTIONS = GAME_CONSTANTS.DIRECTIONS;
// create a new agent
const agent = new kit.Agent();

// first initialize the agent, and then proceed to go in a loop waiting for updates and running the AI
agent.initialize().then(async () => {
  while (true) {
    // wait for update from match engine
    await agent.update();

    // player is your player, opponent is the opposing player
    const player = agent.players[agent.id];
    const opponent = agent.players[(agent.id + 1) % 2];

    /** AI Code goes here */

    let commands = [];

    player.units.forEach((unit) => {
      let allDirs = [
        DIRECTIONS.NORTH,
        DIRECTIONS.EAST,
        DIRECTIONS.SOUTH,
        DIRECTIONS.WEST,
      ];
      commands.push(unit.move(allDirs[Math.floor(Math.random() * 4)]));
    });

    /** AI Code ends here */

    // submit commands to the engine
    console.log(commands.join(','));

    // now we end our turn
    agent.endTurn();

  }
});