import { factory } from '../../utils/factory';
import { deepMap } from '../../utils/collection';
import { cotNumber } from '../../plain/number';
var name = 'cot';
var dependencies = ['typed', 'BigNumber'];
export var createCot = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      _BigNumber = _ref.BigNumber;

  /**
   * Calculate the cotangent of a value. Defined as `cot(x) = 1 / tan(x)`.
   *
   * For matrices, the function is evaluated element wise.
   *
   * Syntax:
   *
   *    math.cot(x)
   *
   * Examples:
   *
   *    math.cot(2)      // returns number -0.45765755436028577
   *    1 / math.tan(2)  // returns number -0.45765755436028577
   *
   * See also:
   *
   *    tan, sec, csc
   *
   * @param {number | Complex | Unit | Array | Matrix} x  Function input
   * @return {number | Complex | Array | Matrix} Cotangent of x
   */
  var cot = typed(name, {
    number: cotNumber,
    Complex: function Complex(x) {
      return x.cot();
    },
    BigNumber: function BigNumber(x) {
      return new _BigNumber(1).div(x.tan());
    },
    Unit: function Unit(x) {
      if (!x.hasBase(x.constructor.BASE_UNITS.ANGLE)) {
        throw new TypeError('Unit in function cot is no angle');
      }

      return cot(x.value);
    },
    'Array | Matrix': function ArrayMatrix(x) {
      return deepMap(x, cot);
    }
  });
  return cot;
});