import { factory } from '../../utils/factory';
import { deepMap } from '../../utils/collection';
var name = 'conj';
var dependencies = ['typed'];
export var createConj = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed;

  /**
   * Compute the complex conjugate of a complex value.
   * If `x = a+bi`, the complex conjugate of `x` is `a - bi`.
   *
   * For matrices, the function is evaluated element wise.
   *
   * Syntax:
   *
   *    math.conj(x)
   *
   * Examples:
   *
   *    math.conj(math.complex('2 + 3i'))  // returns Complex 2 - 3i
   *    math.conj(math.complex('2 - 3i'))  // returns Complex 2 + 3i
   *    math.conj(math.complex('-5.2i'))  // returns Complex 5.2i
   *
   * See also:
   *
   *    re, im, arg, abs
   *
   * @param {number | BigNumber | Complex | Array | Matrix} x
   *            A complex number or array with complex numbers
   * @return {number | BigNumber | Complex | Array | Matrix}
   *            The complex conjugate of x
   */
  var conj = typed(name, {
    number: function number(x) {
      return x;
    },
    BigNumber: function BigNumber(x) {
      return x;
    },
    Complex: function Complex(x) {
      return x.conjugate();
    },
    'Array | Matrix': function ArrayMatrix(x) {
      return deepMap(x, conj);
    }
  });
  return conj;
});