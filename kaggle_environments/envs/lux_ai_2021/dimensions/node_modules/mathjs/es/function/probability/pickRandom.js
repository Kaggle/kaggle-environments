import { factory } from '../../utils/factory';
import { isNumber } from '../../utils/is';
import { arraySize } from '../../utils/array';
import { createRng } from './util/seededRNG';
var name = 'pickRandom';
var dependencies = ['typed', 'config', '?on'];
export var createPickRandom = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      config = _ref.config,
      on = _ref.on;
  // seeded pseudo random number generator
  var rng = createRng(config.randomSeed);

  if (on) {
    on('config', function (curr, prev) {
      if (curr.randomSeed !== prev.randomSeed) {
        rng = createRng(curr.randomSeed);
      }
    });
  }
  /**
   * Random pick one or more values from a one dimensional array.
   * Array elements are picked using a random function with uniform or weighted distribution.
   *
   * Syntax:
   *
   *     math.pickRandom(array)
   *     math.pickRandom(array, number)
   *     math.pickRandom(array, weights)
   *     math.pickRandom(array, number, weights)
   *     math.pickRandom(array, weights, number)
   *
   * Examples:
   *
   *     math.pickRandom([3, 6, 12, 2])                  // returns one of the values in the array
   *     math.pickRandom([3, 6, 12, 2], 2)               // returns an array of two of the values in the array
   *     math.pickRandom([3, 6, 12, 2], [1, 3, 2, 1])    // returns one of the values in the array with weighted distribution
   *     math.pickRandom([3, 6, 12, 2], 2, [1, 3, 2, 1]) // returns an array of two of the values in the array with weighted distribution
   *     math.pickRandom([3, 6, 12, 2], [1, 3, 2, 1], 2) // returns an array of two of the values in the array with weighted distribution
   *
   * See also:
   *
   *     random, randomInt
   *
   * @param {Array | Matrix} array     A one dimensional array
   * @param {Int} number               An int or float
   * @param {Array | Matrix} weights   An array of ints or floats
   * @return {number | Array}          Returns a single random value from array when number is 1 or undefined.
   *                                   Returns an array with the configured number of elements when number is > 1.
   */


  return typed({
    'Array | Matrix': function ArrayMatrix(possibles) {
      return _pickRandom(possibles);
    },
    'Array | Matrix, number': function ArrayMatrixNumber(possibles, number) {
      return _pickRandom(possibles, number, undefined);
    },
    'Array | Matrix, Array': function ArrayMatrixArray(possibles, weights) {
      return _pickRandom(possibles, undefined, weights);
    },
    'Array | Matrix, Array | Matrix, number': function ArrayMatrixArrayMatrixNumber(possibles, weights, number) {
      return _pickRandom(possibles, number, weights);
    },
    'Array | Matrix, number, Array | Matrix': function ArrayMatrixNumberArrayMatrix(possibles, number, weights) {
      return _pickRandom(possibles, number, weights);
    }
  });

  function _pickRandom(possibles, number, weights) {
    var single = typeof number === 'undefined';

    if (single) {
      number = 1;
    }

    possibles = possibles.valueOf(); // get Array

    if (weights) {
      weights = weights.valueOf(); // get Array
    }

    if (arraySize(possibles).length > 1) {
      throw new Error('Only one dimensional vectors supported');
    }

    var totalWeights = 0;

    if (typeof weights !== 'undefined') {
      if (weights.length !== possibles.length) {
        throw new Error('Weights must have the same length as possibles');
      }

      for (var i = 0, len = weights.length; i < len; i++) {
        if (!isNumber(weights[i]) || weights[i] < 0) {
          throw new Error('Weights must be an array of positive numbers');
        }

        totalWeights += weights[i];
      }
    }

    var length = possibles.length;

    if (length === 0) {
      return [];
    } else if (number >= length) {
      return number > 1 ? possibles : possibles[0];
    }

    var result = [];
    var pick;

    while (result.length < number) {
      if (typeof weights === 'undefined') {
        pick = possibles[Math.floor(rng() * length)];
      } else {
        var randKey = rng() * totalWeights;

        for (var _i = 0, _len = possibles.length; _i < _len; _i++) {
          randKey -= weights[_i];

          if (randKey < 0) {
            pick = possibles[_i];
            break;
          }
        }
      }

      if (result.indexOf(pick) === -1) {
        result.push(pick);
      }
    }

    return single ? result[0] : result; // TODO: return matrix when input was a matrix
    // TODO: add support for multi dimensional matrices
  }
});