function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { _defineProperty(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

import { factory } from '../../utils/factory';
import { deepMap } from '../../utils/collection';
import { isInteger } from '../../utils/number';
import { createAlgorithm11 } from '../../type/matrix/utils/algorithm11';
import { createAlgorithm12 } from '../../type/matrix/utils/algorithm12';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
import { roundNumber } from '../../plain/number';
var NO_INT = 'Number of decimals in function round must be an integer';
var name = 'round';
var dependencies = ['typed', 'matrix', 'equalScalar', 'zeros', 'BigNumber', 'DenseMatrix'];
export var createRound = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix,
      equalScalar = _ref.equalScalar,
      zeros = _ref.zeros,
      BigNumber = _ref.BigNumber,
      DenseMatrix = _ref.DenseMatrix;
  var algorithm11 = createAlgorithm11({
    typed: typed,
    equalScalar: equalScalar
  });
  var algorithm12 = createAlgorithm12({
    typed: typed,
    DenseMatrix: DenseMatrix
  });
  var algorithm14 = createAlgorithm14({
    typed: typed
  });
  /**
   * Round a value towards the nearest integer.
   * For matrices, the function is evaluated element wise.
   *
   * Syntax:
   *
   *    math.round(x)
   *    math.round(x, n)
   *
   * Examples:
   *
   *    math.round(3.2)              // returns number 3
   *    math.round(3.8)              // returns number 4
   *    math.round(-4.2)             // returns number -4
   *    math.round(-4.7)             // returns number -5
   *    math.round(math.pi, 3)       // returns number 3.142
   *    math.round(123.45678, 2)     // returns number 123.46
   *
   *    const c = math.complex(3.2, -2.7)
   *    math.round(c)                // returns Complex 3 - 3i
   *
   *    math.round([3.2, 3.8, -4.7]) // returns Array [3, 4, -5]
   *
   * See also:
   *
   *    ceil, fix, floor
   *
   * @param  {number | BigNumber | Fraction | Complex | Array | Matrix} x  Number to be rounded
   * @param  {number | BigNumber | Array} [n=0]                            Number of decimals
   * @return {number | BigNumber | Fraction | Complex | Array | Matrix} Rounded value
   */

  var round = typed(name, _objectSpread(_objectSpread({}, roundNumberSignatures), {}, {
    Complex: function Complex(x) {
      return x.round();
    },
    'Complex, number': function ComplexNumber(x, n) {
      if (n % 1) {
        throw new TypeError(NO_INT);
      }

      return x.round(n);
    },
    'Complex, BigNumber': function ComplexBigNumber(x, n) {
      if (!n.isInteger()) {
        throw new TypeError(NO_INT);
      }

      var _n = n.toNumber();

      return x.round(_n);
    },
    'number, BigNumber': function numberBigNumber(x, n) {
      if (!n.isInteger()) {
        throw new TypeError(NO_INT);
      }

      return new BigNumber(x).toDecimalPlaces(n.toNumber());
    },
    BigNumber: function BigNumber(x) {
      return x.toDecimalPlaces(0);
    },
    'BigNumber, BigNumber': function BigNumberBigNumber(x, n) {
      if (!n.isInteger()) {
        throw new TypeError(NO_INT);
      }

      return x.toDecimalPlaces(n.toNumber());
    },
    Fraction: function Fraction(x) {
      return x.round();
    },
    'Fraction, number': function FractionNumber(x, n) {
      if (n % 1) {
        throw new TypeError(NO_INT);
      }

      return x.round(n);
    },
    'Array | Matrix': function ArrayMatrix(x) {
      // deep map collection, skip zeros since round(0) = 0
      return deepMap(x, round, true);
    },
    'SparseMatrix, number | BigNumber': function SparseMatrixNumberBigNumber(x, y) {
      return algorithm11(x, y, round, false);
    },
    'DenseMatrix, number | BigNumber': function DenseMatrixNumberBigNumber(x, y) {
      return algorithm14(x, y, round, false);
    },
    'number | Complex | BigNumber, SparseMatrix': function numberComplexBigNumberSparseMatrix(x, y) {
      // check scalar is zero
      if (equalScalar(x, 0)) {
        // do not execute algorithm, result will be a zero matrix
        return zeros(y.size(), y.storage());
      }

      return algorithm12(y, x, round, true);
    },
    'number | Complex | BigNumber, DenseMatrix': function numberComplexBigNumberDenseMatrix(x, y) {
      // check scalar is zero
      if (equalScalar(x, 0)) {
        // do not execute algorithm, result will be a zero matrix
        return zeros(y.size(), y.storage());
      }

      return algorithm14(y, x, round, true);
    },
    'Array, number | BigNumber': function ArrayNumberBigNumber(x, y) {
      // use matrix implementation
      return algorithm14(matrix(x), y, round, false).valueOf();
    },
    'number | Complex | BigNumber, Array': function numberComplexBigNumberArray(x, y) {
      // use matrix implementation
      return algorithm14(matrix(y), x, round, true).valueOf();
    }
  }));
  return round;
});
var roundNumberSignatures = {
  number: roundNumber,
  'number, number': function numberNumber(x, n) {
    if (!isInteger(n)) {
      throw new TypeError(NO_INT);
    }

    if (n < 0 || n > 15) {
      throw new Error('Number of decimals in function round must be in te range of 0-15');
    }

    return roundNumber(x, n);
  }
};
export var createRoundNumber = /* #__PURE__ */factory(name, ['typed'], function (_ref2) {
  var typed = _ref2.typed;
  return typed(name, roundNumberSignatures);
});