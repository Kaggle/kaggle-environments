import { errorTransform } from './utils/errorTransform';
import { factory } from '../../utils/factory';
import { createColumn } from '../../function/matrix/column';
import { isNumber } from '../../utils/is';
var name = 'column';
var dependencies = ['typed', 'Index', 'matrix', 'range'];
/**
 * Attach a transform function to matrix.column
 * Adds a property transform containing the transform function.
 *
 * This transform changed the last `index` parameter of function column
 * from zero-based to one-based
 */

export var createColumnTransform = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      Index = _ref.Index,
      matrix = _ref.matrix,
      range = _ref.range;
  var column = createColumn({
    typed: typed,
    Index: Index,
    matrix: matrix,
    range: range
  }); // @see: comment of column itself

  return typed('column', {
    '...any': function any(args) {
      // change last argument from zero-based to one-based
      var lastIndex = args.length - 1;
      var last = args[lastIndex];

      if (isNumber(last)) {
        args[lastIndex] = last - 1;
      }

      try {
        return column.apply(null, args);
      } catch (err) {
        throw errorTransform(err);
      }
    }
  });
}, {
  isTransformFunction: true
});