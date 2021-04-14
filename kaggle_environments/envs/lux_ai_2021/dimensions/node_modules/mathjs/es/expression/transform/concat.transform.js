import { isBigNumber, isNumber } from '../../utils/is';
import { errorTransform } from './utils/errorTransform';
import { factory } from '../../utils/factory';
import { createConcat } from '../../function/matrix/concat';
var name = 'concat';
var dependencies = ['typed', 'matrix', 'isInteger'];
export var createConcatTransform = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix,
      isInteger = _ref.isInteger;
  var concat = createConcat({
    typed: typed,
    matrix: matrix,
    isInteger: isInteger
  });
  /**
   * Attach a transform function to math.range
   * Adds a property transform containing the transform function.
   *
   * This transform changed the last `dim` parameter of function concat
   * from one-based to zero based
   */

  return typed('concat', {
    '...any': function any(args) {
      // change last argument from one-based to zero-based
      var lastIndex = args.length - 1;
      var last = args[lastIndex];

      if (isNumber(last)) {
        args[lastIndex] = last - 1;
      } else if (isBigNumber(last)) {
        args[lastIndex] = last.minus(1);
      }

      try {
        return concat.apply(null, args);
      } catch (err) {
        throw errorTransform(err);
      }
    }
  });
}, {
  isTransformFunction: true
});