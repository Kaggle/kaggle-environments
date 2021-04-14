import { factory } from '../../utils/factory';
import { createRow } from '../../function/matrix/row';
import { errorTransform } from './utils/errorTransform';
import { isNumber } from '../../utils/is';
var name = 'row';
var dependencies = ['typed', 'Index', 'matrix', 'range'];
/**
 * Attach a transform function to matrix.column
 * Adds a property transform containing the transform function.
 *
 * This transform changed the last `index` parameter of function column
 * from zero-based to one-based
 */

export var createRowTransform = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      Index = _ref.Index,
      matrix = _ref.matrix,
      range = _ref.range;
  var row = createRow({
    typed: typed,
    Index: Index,
    matrix: matrix,
    range: range
  }); // @see: comment of row itself

  return typed('row', {
    '...any': function any(args) {
      // change last argument from zero-based to one-based
      var lastIndex = args.length - 1;
      var last = args[lastIndex];

      if (isNumber(last)) {
        args[lastIndex] = last - 1;
      }

      try {
        return row.apply(null, args);
      } catch (err) {
        throw errorTransform(err);
      }
    }
  });
}, {
  isTransformFunction: true
});