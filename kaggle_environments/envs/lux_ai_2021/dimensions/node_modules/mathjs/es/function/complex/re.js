import { factory } from '../../utils/factory';
import { deepMap } from '../../utils/collection';
var name = 're';
var dependencies = ['typed'];
export var createRe = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed;

  /**
   * Get the real part of a complex number.
   * For a complex number `a + bi`, the function returns `a`.
   *
   * For matrices, the function is evaluated element wise.
   *
   * Syntax:
   *
   *    math.re(x)
   *
   * Examples:
   *
   *    const a = math.complex(2, 3)
   *    math.re(a)                     // returns number 2
   *    math.im(a)                     // returns number 3
   *
   *    math.re(math.complex('-5.2i')) // returns number 0
   *    math.re(math.complex(2.4))     // returns number 2.4
   *
   * See also:
   *
   *    im, conj, abs, arg
   *
   * @param {number | BigNumber | Complex | Array | Matrix} x
   *            A complex number or array with complex numbers
   * @return {number | BigNumber | Array | Matrix} The real part of x
   */
  var re = typed(name, {
    number: function number(x) {
      return x;
    },
    BigNumber: function BigNumber(x) {
      return x;
    },
    Complex: function Complex(x) {
      return x.re;
    },
    'Array | Matrix': function ArrayMatrix(x) {
      return deepMap(x, re);
    }
  });
  return re;
});