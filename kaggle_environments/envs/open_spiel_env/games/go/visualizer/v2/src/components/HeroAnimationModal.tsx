import { memo } from 'react';
import { Game } from 'tenuki';
import useGameStore from '../stores/useGameStore';

export default memo(function HeroAnimationModal() {
  const game = useGameStore((state) => state.game);

  function isLadder(game: Game) {
    const history = game._moves;

    if (history.length < 9) {
      return false;
    }

    if (
      history.at(-1).pass ||
      history.at(-3).pass ||
      history.at(-5).pass ||
      history.at(-7).pass ||
      history.at(-9).pass
    ) {
      return false;
    }

    const d1y = history.at(-3).playedPoint.y - history.at(-1).playedPoint.y;
    const d1x = history.at(-3).playedPoint.x - history.at(-1).playedPoint.x;
    const d2y = history.at(-5).playedPoint.y - history.at(-3).playedPoint.y;
    const d2x = history.at(-5).playedPoint.x - history.at(-3).playedPoint.x;

    if (d1y !== 0 && d1x !== 0) {
      return false;
    }

    if (d2y !== 0 && d2x !== 0) {
      return false;
    }

    if (Math.abs(d1y + d2y) !== 1 || Math.abs(d1x + d2x) !== 1) {
      return false;
    }

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

    return true;
  }

  const isMonkeyJump = (game: Game) => {
    const history = game._moves;

    if (history.length < 5) {
      return false;
    }

    if (history.at(-1).pass) {
      return false;
    }

    const state = game.currentState();
    const point = state.playedPoint!;
    const color = state.color;
    const max = game.boardSize - 1;

    if (point.x !== 0 && point.y !== 0 && point.x !== max && point.y !== max) {
      return false;
    }

    if (
      (point.x < 3 && point.y < 3) ||
      (point.x < 3 && point.y > max - 3) ||
      (point.x > max - 3 && point.y < 3) ||
      (point.x > max - 3 && point.y > max - 3)
    ) {
      return false;
    }

    let dy, dx, ey, ex;
    if (point.x === 0 && point.y <= max / 2) [dy, dx, ey, ex] = [0, 1, 0, -1];
    if (point.x === 0 && point.y > max / 2) [dy, dx, ey, ex] = [0, 1, 0, 1];
    if (point.y === 0 && point.x <= max / 2) [dy, dx, ey, ex] = [1, 0, -1, 0];
    if (point.y === 0 && point.x > max / 2) [dy, dx, ey, ex] = [1, 0, 1, 0];
    if (point.x === max && point.y <= max / 2) [dy, dx, ey, ex] = [0, -1, 0, -1];
    if (point.x === max && point.y > max / 2) [dy, dx, ey, ex] = [0, -1, 0, 1];
    if (point.y === max && point.x <= max / 2) [dy, dx, ey, ex] = [-1, 0, -1, 0];
    if (point.y === max && point.x > max / 2) [dy, dx, ey, ex] = [-1, 0, 1, 0];

    let neighbors = [];
    neighbors.push(...state.neighborsFor(point.y + dx! * 2, point.x + dy! * 2));
    neighbors.push(...state.neighborsFor(point.y + dx!, point.x + dy!));
    neighbors.push(...state.neighborsFor(point.y, point.x));
    neighbors.push(...state.neighborsFor(point.y - dx!, point.x - dy!));
    neighbors.push(...state.neighborsFor(point.y - dx! * 2, point.x - dy! * 2));
    neighbors = [...new Set(neighbors)];
    neighbors = neighbors.filter((intersection) => intersection.isEmpty());
    if (neighbors.length !== 11) {
      return false;
    }

    const sameColor = state
      .groupAt(point.y + ey! - 3 * ex!, point.x - 3 * ey! + ex!)
      .filter((intersection) => intersection.isOccupiedWith(color) === true);
    if (sameColor.length < 2) {
      return false;
    }

    const oppColor = state
      .groupAt(point.y + 2 * ex! + 2 * ey!, point.x - 2 * ex! - 2 * ey!)
      .filter((intersection) => intersection.isOccupiedWith(color) === false && intersection.isEmpty() === false);
    if (oppColor.length < 2) {
      return false;
    }

    return true;
  };

  const state = game.currentState();

  const isPass = state.pass === true;
  const isDoublePass = state.pass === true && game._moves.at(-2).pass === true;
  const isLossOfADragon = state.capturedPositions !== undefined && state.capturedPositions.length > 6;

  console.log('hero animation', isLadder(game), isPass, isDoublePass, isLossOfADragon, isMonkeyJump(game));

  return null;
});
