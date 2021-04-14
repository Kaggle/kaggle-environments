import { isBigNumber, isCollection, isNumber } from '../../utils/is';
import { factory } from '../../utils/factory';
import { errorTransform } from './utils/errorTransform';
import { createMean } from '../../function/statistics/mean';
var name = 'mean';
var dependencies = ['typed', 'add', 'divide'];
export var createMeanTransform = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      add = _ref.add,
      divide = _ref.divide;
  var mean = createMean({
    typed: typed,
    add: add,
    divide: divide
  });
  /**
   * Attach a transform function to math.mean
   * Adds a property transform containing the transform function.
   *
   * This transform changed the last `dim` parameter of function mean
   * from one-based to zero based
   */

  return typed('mean', {
    '...any': function any(args) {
      // change last argument dim from one-based to zero-based
      if (args.length === 2 && isCollection(args[0])) {
        var dim = args[1];

        if (isNumber(dim)) {
          args[1] = dim - 1;
        } else if (isBigNumber(dim)) {
          args[1] = dim.minus(1);
        }
      }

      try {
        return mean.apply(null, args);
      } catch (err) {
        throw errorTransform(err);
      }
    }
  });
}, {
  isTransformFunction: true
});