import { factory } from '../../utils/factory';
import { deepMap } from '../../utils/collection';
import { unaryMinusNumber } from '../../plain/number';
var name = 'unaryMinus';
var dependencies = ['typed'];
export var createUnaryMinus = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed;

  /**
   * Inverse the sign of a value, apply a unary minus operation.
   *
   * For matrices, the function is evaluated element wise. Boolean values and
   * strings will be converted to a number. For complex numbers, both real and
   * complex value are inverted.
   *
   * Syntax:
   *
   *    math.unaryMinus(x)
   *
   * Examples:
   *
   *    math.unaryMinus(3.5)      // returns -3.5
   *    math.unaryMinus(-4.2)     // returns 4.2
   *
   * See also:
   *
   *    add, subtract, unaryPlus
   *
   * @param  {number | BigNumber | Fraction | Complex | Unit | Array | Matrix} x Number to be inverted.
   * @return {number | BigNumber | Fraction | Complex | Unit | Array | Matrix} Returns the value with inverted sign.
   */
  var unaryMinus = typed(name, {
    number: unaryMinusNumber,
    Complex: function Complex(x) {
      return x.neg();
    },
    BigNumber: function BigNumber(x) {
      return x.neg();
    },
    Fraction: function Fraction(x) {
      return x.neg();
    },
    Unit: function Unit(x) {
      var res = x.clone();
      res.value = unaryMinus(x.value);
      return res;
    },
    'Array | Matrix': function ArrayMatrix(x) {
      // deep map collection, skip zeros since unaryMinus(0) = 0
      return deepMap(x, unaryMinus, true);
    } // TODO: add support for string

  });
  return unaryMinus;
});