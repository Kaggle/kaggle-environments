const {
  closestDirectionToPos,
  findClosestPos,
  hashPos,
  ROLES,
} = require('./utils');
const {
  Position
} = require('./lux/map');
const map = require('./lux/map');
const kit = require('./lux/kit');

const deltas = {
  1: [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
  ]
}

const runworker = (agent, unit, state) => {
  const commands = [];
  let moved = false;
  const forestResources = agent.resources;
  const cityTilesArr = agent.cityTilesArr;
  const spotsTaken = agent.spotsTaken;
  // go to nearest forest, mine until just enough time to get back to city or full
  if (state.role === ROLES.BUILDCITY) {

    // check if self is adjacent to a existing city;
    // agent.map.getCellByPos(unit.pos).resource === null && agent.map.getCellByPos(unit.pos).cityTile === null
    let closestAdjacentTile = null;
    let closestDist = 9999999;
    if (cityTilesArr.length) {

      let closestTile = findClosestPos(unit.pos, cityTilesArr.map((ct) => ct.pos));

      for (const delta of deltas[1]) {
        let x = closestTile.x;
        let y = closestTile.y;
        const adj = new Position(x + delta[0], y + delta[1]);
        if (adj.x < 0 || adj.y < 0 || adj.y >= agent.map.height || adj.x >= agent.map.width) {
          continue;
        }
        const cell = agent.map.getCellByPos(adj);
        if (cell.resource === null && cell.citytile === null) {
          let dist = adj.distanceTo(unit.pos);
          if (dist < closestDist) {
            closestDist = dist;
            closestAdjacentTile = adj;
          }
        }
      }
    }
    if (closestAdjacentTile) {
      if (closestAdjacentTile.equals(unit.pos)) {
        commands.push(unit.buildCity());
        state.role = ROLES.MINE;
      } else {
        const dir = closestDirectionToPos(unit.pos, closestAdjacentTile, spotsTaken);
        // const dir = unit.pos.directionTo(closestForestPos);
        if (dir !== null) {
          commands.push(unit.move(dir));
          spotsTaken.add(hashPos(unit.pos.translate(dir, 1)));
          moved = true;
        }
      }
    }
  } else if (state.role === ROLES.MINE) {

    let closestForestPos = null;
    if (forestResources.length) {
      closestForestPos = findClosestPos(unit.pos, forestResources.map(({
        pos
      }) => pos));
      for (let i = 0; i < forestResources.length; i++) {
        const r = forestResources[i];
        if (r.pos.equals(closestForestPos)) {
          forestResources.splice(i, 1);
          break;
        }
      }
    }
    if (closestForestPos) {
      
      let dir = closestDirectionToPos(unit.pos, closestForestPos, spotsTaken);
      // const dir = unit.pos.directionTo(closestForestPos);
      if (closestForestPos.equals(unit.pos)) {
        dir = null;
      }
      if (dir !== null) {
        commands.push(kit.annotate.line(unit.pos.x, unit.pos.y, closestForestPos.x, closestForestPos.y))
        commands.push(unit.move(dir));
        spotsTaken.add(hashPos(unit.pos.translate(dir, 1)));
        moved = true;
      }
    }

  } else if (state.role === ROLES.RETURN) {
    let closestCityPos = null;
    if (cityTilesArr.length) {
      closestCityPos = findClosestPos(unit.pos, cityTilesArr.map((cityTile) => cityTile.pos));
    }
    if (closestCityPos) {
      const dir = closestDirectionToPos(unit.pos, closestCityPos, spotsTaken);
      // const dir = unit.pos.directionTo(closestForestPos);
      if (dir !== null) {
        commands.push(unit.move(dir));
        spotsTaken.add(hashPos(unit.pos.translate(dir, 1)));
        moved = true;
        
      }
    }
  }
  if (!moved) {
    spotsTaken.add(hashPos(unit.pos));
  }
  return commands;
};

module.exports = {
  runworker,
}