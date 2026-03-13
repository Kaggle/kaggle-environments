import { memo } from 'react';
import { Game, Intersection } from 'tenuki';
import useGameStore from '../stores/useGameStore';

export default memo(function HeroAnimationModal() {
  const game = useGameStore((state) => state.game);

  function isLadder(game: Game) {
    const history = game._moves;

    // 
    if (history.length < 9) return false;

    // Looking at the history of moves so first check they aren't passes
    if (
      history.at(-1).pass ||
      history.at(-3).pass ||
      history.at(-5).pass ||
      history.at(-7).pass ||
      history.at(-9).pass
    ) {
      return false;
    }

    // Each move should be next to the previous
    const d1y = history.at(-3).playedPoint.y - history.at(-1).playedPoint.y;
    const d1x = history.at(-3).playedPoint.x - history.at(-1).playedPoint.x;

    if (d1y !== 0 && d1x !== 0) return false;
    if (Math.abs(d1x) + Math.abs(d1y) !== 1) return false;

    // Not be in the same direction as the previous
    const d2y = history.at(-5).playedPoint.y - history.at(-3).playedPoint.y;
    const d2x = history.at(-5).playedPoint.x - history.at(-3).playedPoint.x;

    if (d2y !== 0 && d2x !== 0) return false;
    if (Math.abs(d2x) + Math.abs(d2y) !== 1) return false;
    if (d1x === d2x && d1y === d2y) return false;

    // Moves should step in the same directions
    if (
      history.at(-7).playedPoint.y !== history.at(-5).playedPoint.y + d1y ||
      history.at(-7).playedPoint.x !== history.at(-5).playedPoint.x + d1x
    ) {
      return false;
    }
    if (
      history.at(-9).playedPoint.y !== history.at(-7).playedPoint.y + d2y ||
      history.at(-9).playedPoint.x !== history.at(-7).playedPoint.x + d2x
    ) {
      return false;
    }

    // Three rungs of the ladder so a group of six stones
    const point = history.at(-1).playedPoint;
    const group: Intersection[] = history.at(-1).groupAt(point.y, point.x);

    if (group.length !== 6) return false;

    // After the opponents previous move five of the stones in atari
    if (group.filter((intersection) => history.at(-2).inAtari(intersection.y, intersection.x)).length !== 5) {
      return false;
    }

    return true;
  }

  const isMonkeyJump = (game: Game) => {
    const state = game.currentState();

    if (!state.playedPoint) return false;

    const point = state.intersectionAt(state.playedPoint.y, state.playedPoint.x);
    const color = state.color;
    const max = game.boardSize - 1;

    // Played stone is on an edge and not in a corner
    let dy, dx;
    switch (true) {
      case point.x === 0 && point.y >= 3 && point.y <= max - 3:
        dy = 0;
        dx = 1;
        break;
      case point.y === max && point.x >= 3 && point.x <= max - 3:
        dy = -1;
        dx = 0;
        break;
      case point.x === max && point.y >= 3 && point.y <= max - 3:
        dy = 0;
        dx = -1;
        break;
      case point.y === 0 && point.x >= 3 && point.x <= max - 3:
        dy = 1;
        dx = 0;
        break;
      default:
        return false;
    }

    // Space around played stone
    const emptyNeighbors = [
      ...state.neighborsFor(point.y + 2 * dx, point.x + 2 * dy),
      ...state.neighborsFor(point.y + dx, point.x + dy),
      ...state.neighborsFor(point.y, point.x),
      ...state.neighborsFor(point.y - dx, point.x - dy),
      ...state.neighborsFor(point.y - 2 * dx, point.x - 2 * dy),
    ].filter((intersection) => intersection.isEmpty());

    if ([...new Set(emptyNeighbors)].length !== 11) return false;

    // One of two possible positions for a stone from the same player
    const possibleSameColorStones = [
      state.intersectionAt(point.y + dy - 3 * Math.abs(dx), point.x + dx - 3 * Math.abs(dy)),
      state.intersectionAt(point.y + dy + 3 * Math.abs(dx), point.x + dx + 3 * Math.abs(dy)),
    ].filter((intersection) => intersection.isEmpty() === false);

    if (possibleSameColorStones.length !== 1) return false;

    const sameColorStone = possibleSameColorStones.at(0)!;

    if (sameColorStone.isOccupiedWith(color) === false) return false;

    // Opponents stone matching the direction as the same color found stone
    const direction = sameColorStone.x * dy + sameColorStone.y * dx < point.x * dy + point.y * dx ? -1 : 1;
    const opponentColorStone = state.intersectionAt(
      point.y + 2 * (dy + direction * dx),
      point.x + 2 * (dx + direction * dy)
    );

    if (opponentColorStone.isOccupiedWith(color) || opponentColorStone.isEmpty()) return false;

    // Each found stone should be part of a group, let's say two or more of that color
    if (state.groupAt(sameColorStone.y, sameColorStone.x).length < 2) return false;
    if (state.groupAt(opponentColorStone.y, opponentColorStone.x).length < 2) return false;

    return true;
  };

  const state = game.currentState();

  const isPass = state.pass === true;
  const isDoublePass = state.pass === true && game._moves.at(-2).pass === true;
  const isLossOfADragon = state.capturedPositions !== undefined && state.capturedPositions.length > 6;

  console.log('hero animation', isLadder(game), isPass, isDoublePass, isLossOfADragon, isMonkeyJump(game));

  return null;
});
