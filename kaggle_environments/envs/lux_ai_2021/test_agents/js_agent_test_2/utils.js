const ROLES = {
  MINE: 'mine',
  BUILDCITY: 'bc',
  RETURN: 'return',
}
const GAME_CONSTANTS = require('./lux/game_constants');
const DIRECTIONS = GAME_CONSTANTS.DIRECTIONS;

const findClosestPos = (startPos, arrOfPositions) => {
  let closestPos = arrOfPositions[0];
  let closestDist = 9999999;
  arrOfPositions.forEach((pos) => {
    const dist = startPos.distanceTo(pos);
    if (dist < closestDist) {
      closestDist = dist;
      closestPos = pos;
    }
  });
  return closestPos;
};
const hashPos = (pos) => {
  return pos.x * 1000 + pos.y;
};

const closestDirectionToPos = (start, end, spotsTaken) => {
  const checkDirections = [
    DIRECTIONS.NORTH,
    DIRECTIONS.EAST,
    DIRECTIONS.SOUTH,
    DIRECTIONS.WEST,
  ];
  let closestDirection = null;
  let closestDist = start.distanceTo(end);
  for (let i = 0; i < checkDirections.length; i++) {
    const dir = checkDirections[i];
    const newpos = start.translate(dir, 1);
    if (!spotsTaken.has(hashPos(newpos))) {
      const dist = end.distanceTo(newpos);
      if (dist < closestDist) {
        closestDist = dist;
        closestDirection = dir;
      }
    }
  }
  return closestDirection;
}

module.exports = {
  findClosestPos,
  hashPos,
  ROLES,
  closestDirectionToPos,
}