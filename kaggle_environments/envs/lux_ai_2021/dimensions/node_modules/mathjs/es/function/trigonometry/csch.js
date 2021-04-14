import { factory } from '../../utils/factory';
import { deepMap } from '../../utils/collection';
import { cschNumber } from '../../plain/number';
var name = 'csch';
var dependencies = ['typed', 'BigNumber'];
export var createCsch = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      _BigNumber = _ref.BigNumber;

  /**
   * Calculate the hyperbolic cosecant of a value,
   * defined as `csch(x) = 1 / sinh(x)`.
   *
   * For matrices, the function is evaluated element wise.
   *
   * Syntax:
   *
   *    math.csch(x)
   *
   * Examples:
   *
   *    // csch(x) = 1/ sinh(x)
   *    math.csch(0.5)       // returns 1.9190347513349437
   *    1 / math.sinh(0.5)   // returns 1.9190347513349437
   *
   * See also:
   *
   *    sinh, sech, coth
   *
   * @param {number | Complex | Unit | Array | Matrix} x  Function input
   * @return {number | Complex | Array | Matrix} Hyperbolic cosecant of x
   */
  var csch = typed(name, {
    number: cschNumber,
    Complex: function Complex(x) {
      return x.csch();
    },
    BigNumber: function BigNumber(x) {
      return new _BigNumber(1).div(x.sinh());
    },
    Unit: function Unit(x) {
      if (!x.hasBase(x.constructor.BASE_UNITS.ANGLE)) {
        throw new TypeError('Unit in function csch is no angle');
      }

      return csch(x.value);
    },
    'Array | Matrix': function ArrayMatrix(x) {
      return deepMap(x, csch);
    }
  });
  return csch;
});