import { factory } from '../../utils/factory';
import { isBigNumber, isCollection, isNumber } from '../../utils/is';
import { errorTransform } from './utils/errorTransform';
import { createVariance } from '../../function/statistics/variance';
var name = 'variance';
var dependencies = ['typed', 'add', 'subtract', 'multiply', 'divide', 'apply', 'isNaN'];
/**
 * Attach a transform function to math.var
 * Adds a property transform containing the transform function.
 *
 * This transform changed the `dim` parameter of function var
 * from one-based to zero based
 */

export var createVarianceTransform = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      add = _ref.add,
      subtract = _ref.subtract,
      multiply = _ref.multiply,
      divide = _ref.divide,
      apply = _ref.apply,
      isNaN = _ref.isNaN;
  var variance = createVariance({
    typed: typed,
    add: add,
    subtract: subtract,
    multiply: multiply,
    divide: divide,
    apply: apply,
    isNaN: isNaN
  });
  return typed(name, {
    '...any': function any(args) {
      // change last argument dim from one-based to zero-based
      if (args.length >= 2 && isCollection(args[0])) {
        var dim = args[1];

        if (isNumber(dim)) {
          args[1] = dim - 1;
        } else if (isBigNumber(dim)) {
          args[1] = dim.minus(1);
        }
      }

      try {
        return variance.apply(null, args);
      } catch (err) {
        throw errorTransform(err);
      }
    }
  });
}, {
  isTransformFunction: true
});