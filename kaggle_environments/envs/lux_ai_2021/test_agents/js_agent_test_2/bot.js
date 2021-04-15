const kit = require('./lux/kit');
const GAME_CONSTANTS = require('./lux/game_constants');
const {
  Position
} = require('./lux/map');
const {
  runworker
} = require('./worker');
const {
  ROLES
} = require('./utils');
const DIRECTIONS = GAME_CONSTANTS.DIRECTIONS;
// create a new agent
const agent = new kit.Agent();

// first initialize the agent, and then proceed to go in a loop waiting for updates and running the AI
agent.initialize().then(async () => {

  const unitStates = {};

  while (true) {
    // wait for update from match engine
    await agent.update();

    // player is your player, opponent is the opposing player
    const player = agent.players[agent.id];
    const opponent = agent.players[(agent.id + 1) % 2];

    /** AI Code goes here */

    let commands = [];

    agent.resources = [];
    for (let y = 0; y < agent.mapHeight; y++) {
      for (let x = 0; x < agent.mapWidth; x++) {
        const cell = agent.map.getCell(x, y);
        const resource = cell.resource;
        if (resource !== null) {
          if (resource.type === GAME_CONSTANTS.RESOURCE_TYPES.WOOD) {
            agent.resources.push({
              amount: resource.amount,
              pos: new Position(x, y)
            });
          } else if (player.researchedCoal() && resource.type === GAME_CONSTANTS.RESOURCE_TYPES.COAL) {
            agent.resources.push({
              amount: resource.amount,
              pos: new Position(x, y)
            });
          } else if (player.researchedUranium() && resource.type === GAME_CONSTANTS.RESOURCE_TYPES.URANIUM) {
            agent.resources.push({
              amount: resource.amount,
              pos: new Position(x, y)
            });
          }
        }
      }
    }

    agent.cityTilesArr = [];
    agent.spotsTaken = new Set();
    let citiesSafe = 0;
    player.cities.forEach((city) => {
      agent.cityTilesArr.push(...city.citytiles);
      if (city.getLightUpkeep() < city.fuel - 400 && agent.turn % 20 < 10) {
        citiesSafe += 1;
      }
    });
    let workerCount = 0;
    let cartCount = 0;
    // console.error("======== " + agent.turn + " =======");
    // console.error("Cities: ", player.cities.size)

    // figure out roles first, then act
    let numberBuildingCity = 0;
    const workerUnits = [];
    player.units.forEach((unit) => {
      if (unit.isWorker()) {
        workerCount++;
        workerUnits.push(unit);
        if (unitStates[unit.id] === undefined) {
          unitStates[unit.id] = {
            role: ROLES.MINE
          }
        }
        if (unitStates[unit.id].role === ROLES.BUILDCITY) {
          numberBuildingCity++;
          return;
        }
        if (citiesSafe === player.cities.size && numberBuildingCity < 1) {
          unitStates[unit.id].role = ROLES.BUILDCITY;
          numberBuildingCity++;
        } else if (unit.getCargoSpaceLeft() === 0 || unit.cargo.uranium > 3 || unit.cargo.coal > 20) {
          unitStates[unit.id].role = ROLES.RETURN;
        } else {
          unitStates[unit.id].role = ROLES.MINE;
        }
      } else {
        cartCount++;
        // if (agent.turn % 20 < 10) {
        //   if (Math.random() < 0.5) {
        //     commands.push(unit.move('s'));
        //   } else {
        //     commands.push(unit.move('w'));
        //   }
        // } else {
        //   if (Math.random() < 0.5) {
        //     commands.push(unit.move('n'));
        //   } else {
        //     commands.push(unit.move('e'));
        //   }
        // }
      }
    });

    workerUnits.forEach((worker) => {
      let cmds = runworker(agent, worker, unitStates[worker.id]);
      commands.push(...cmds);
    });

    agent.cityTilesArr.forEach((cityTile) => {
      if (cityTile.canAct()) {
        if (agent.cityTilesArr.length > workerCount + cartCount) {
          commands.push(cityTile.buildWorker());
          workerCount++;
        } else {
          commands.push(cityTile.research());
        }
      }
    });

    // commands.push(kit.annotate.circle(7, 7));
    // commands.push(kit.annotate.x(10, 7));
    /** AI Code ends here */

    // submit commands to the engine
    console.log(commands.join(','));



    // now we end our turn
    agent.endTurn();

  }
});