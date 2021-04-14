import { factory } from '../../utils/factory';
import { createStd } from '../../function/statistics/std';
import { isBigNumber, isNumber, isCollection } from '../../utils/is';
import { errorTransform } from './utils/errorTransform';
var name = 'std';
var dependencies = ['typed', 'sqrt', 'variance'];
/**
 * Attach a transform function to math.std
 * Adds a property transform containing the transform function.
 *
 * This transform changed the `dim` parameter of function std
 * from one-based to zero based
 */

export var createStdTransform = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      sqrt = _ref.sqrt,
      variance = _ref.variance;
  var std = createStd({
    typed: typed,
    sqrt: sqrt,
    variance: variance
  });
  return typed('std', {
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
        return std.apply(null, args);
      } catch (err) {
        throw errorTransform(err);
      }
    }
  });
}, {
  isTransformFunction: true
});