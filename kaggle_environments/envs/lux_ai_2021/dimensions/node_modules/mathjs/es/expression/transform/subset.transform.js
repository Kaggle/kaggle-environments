import { factory } from '../../utils/factory';
import { errorTransform } from './utils/errorTransform';
import { createSubset } from '../../function/matrix/subset';
var name = 'subset';
var dependencies = ['typed', 'matrix'];
export var createSubsetTransform = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix;
  var subset = createSubset({
    typed: typed,
    matrix: matrix
  });
  /**
   * Attach a transform function to math.subset
   * Adds a property transform containing the transform function.
   *
   * This transform creates a range which includes the end value
   */

  return typed('subset', {
    '...any': function any(args) {
      try {
        return subset.apply(null, args);
      } catch (err) {
        throw errorTransform(err);
      }
    }
  });
}, {
  isTransformFunction: true
});