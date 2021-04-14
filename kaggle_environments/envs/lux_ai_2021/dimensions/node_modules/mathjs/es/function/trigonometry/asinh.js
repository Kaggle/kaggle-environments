import { factory } from '../../utils/factory';
import { deepMap } from '../../utils/collection';
import { asinhNumber } from '../../plain/number';
var name = 'asinh';
var dependencies = ['typed'];
export var createAsinh = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed;

  /**
   * Calculate the hyperbolic arcsine of a value,
   * defined as `asinh(x) = ln(x + sqrt(x^2 + 1))`.
   *
   * For matrices, the function is evaluated element wise.
   *
   * Syntax:
   *
   *    math.asinh(x)
   *
   * Examples:
   *
   *    math.asinh(0.5)       // returns 0.48121182505960347
   *
   * See also:
   *
   *    acosh, atanh
   *
   * @param {number | Complex | Array | Matrix} x  Function input
   * @return {number | Complex | Array | Matrix} Hyperbolic arcsine of x
   */
  var asinh = typed('asinh', {
    number: asinhNumber,
    Complex: function Complex(x) {
      return x.asinh();
    },
    BigNumber: function BigNumber(x) {
      return x.asinh();
    },
    'Array | Matrix': function ArrayMatrix(x) {
      // deep map collection, skip zeros since asinh(0) = 0
      return deepMap(x, asinh, true);
    }
  });
  return asinh;
});