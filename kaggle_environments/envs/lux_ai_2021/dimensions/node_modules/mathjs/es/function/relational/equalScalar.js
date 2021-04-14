import { nearlyEqual as bigNearlyEqual } from '../../utils/bignumber/nearlyEqual';
import { nearlyEqual } from '../../utils/number';
import { factory } from '../../utils/factory';
import { complexEquals } from '../../utils/complex';
var name = 'equalScalar';
var dependencies = ['typed', 'config'];
export var createEqualScalar = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      config = _ref.config;

  /**
   * Test whether two scalar values are nearly equal.
   *
   * @param  {number | BigNumber | Fraction | boolean | Complex | Unit} x   First value to compare
   * @param  {number | BigNumber | Fraction | boolean | Complex} y          Second value to compare
   * @return {boolean}                                                  Returns true when the compared values are equal, else returns false
   * @private
   */
  var equalScalar = typed(name, {
    'boolean, boolean': function booleanBoolean(x, y) {
      return x === y;
    },
    'number, number': function numberNumber(x, y) {
      return nearlyEqual(x, y, config.epsilon);
    },
    'BigNumber, BigNumber': function BigNumberBigNumber(x, y) {
      return x.eq(y) || bigNearlyEqual(x, y, config.epsilon);
    },
    'Fraction, Fraction': function FractionFraction(x, y) {
      return x.equals(y);
    },
    'Complex, Complex': function ComplexComplex(x, y) {
      return complexEquals(x, y, config.epsilon);
    },
    'Unit, Unit': function UnitUnit(x, y) {
      if (!x.equalBase(y)) {
        throw new Error('Cannot compare units with different base');
      }

      return equalScalar(x.value, y.value);
    }
  });
  return equalScalar;
});
export var createEqualScalarNumber = factory(name, ['typed', 'config'], function (_ref2) {
  var typed = _ref2.typed,
      config = _ref2.config;
  return typed(name, {
    'number, number': function numberNumber(x, y) {
      return nearlyEqual(x, y, config.epsilon);
    }
  });
});