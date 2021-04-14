import { nearlyEqual as bigNearlyEqual } from '../../utils/bignumber/nearlyEqual';
import { nearlyEqual } from '../../utils/number';
import { factory } from '../../utils/factory';
import { createAlgorithm03 } from '../../type/matrix/utils/algorithm03';
import { createAlgorithm07 } from '../../type/matrix/utils/algorithm07';
import { createAlgorithm12 } from '../../type/matrix/utils/algorithm12';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
var name = 'largerEq';
var dependencies = ['typed', 'config', 'matrix', 'DenseMatrix'];
export var createLargerEq = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      config = _ref.config,
      matrix = _ref.matrix,
      DenseMatrix = _ref.DenseMatrix;
  var algorithm03 = createAlgorithm03({
    typed: typed
  });
  var algorithm07 = createAlgorithm07({
    typed: typed,
    DenseMatrix: DenseMatrix
  });
  var algorithm12 = createAlgorithm12({
    typed: typed,
    DenseMatrix: DenseMatrix
  });
  var algorithm13 = createAlgorithm13({
    typed: typed
  });
  var algorithm14 = createAlgorithm14({
    typed: typed
  });
  /**
   * Test whether value x is larger or equal to y.
   *
   * The function returns true when x is larger than y or the relative
   * difference between x and y is smaller than the configured epsilon. The
   * function cannot be used to compare values smaller than approximately 2.22e-16.
   *
   * For matrices, the function is evaluated element wise.
   * Strings are compared by their numerical value.
   *
   * Syntax:
   *
   *    math.largerEq(x, y)
   *
   * Examples:
   *
   *    math.larger(2, 1 + 1)         // returns false
   *    math.largerEq(2, 1 + 1)       // returns true
   *
   * See also:
   *
   *    equal, unequal, smaller, smallerEq, larger, compare
   *
   * @param  {number | BigNumber | Fraction | boolean | Unit | string | Array | Matrix} x First value to compare
   * @param  {number | BigNumber | Fraction | boolean | Unit | string | Array | Matrix} y Second value to compare
   * @return {boolean | Array | Matrix} Returns true when the x is larger or equal to y, else returns false
   */

  var largerEq = typed(name, {
    'boolean, boolean': function booleanBoolean(x, y) {
      return x >= y;
    },
    'number, number': function numberNumber(x, y) {
      return x >= y || nearlyEqual(x, y, config.epsilon);
    },
    'BigNumber, BigNumber': function BigNumberBigNumber(x, y) {
      return x.gte(y) || bigNearlyEqual(x, y, config.epsilon);
    },
    'Fraction, Fraction': function FractionFraction(x, y) {
      return x.compare(y) !== -1;
    },
    'Complex, Complex': function ComplexComplex() {
      throw new TypeError('No ordering relation is defined for complex numbers');
    },
    'Unit, Unit': function UnitUnit(x, y) {
      if (!x.equalBase(y)) {
        throw new Error('Cannot compare units with different base');
      }

      return largerEq(x.value, y.value);
    },
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      return algorithm07(x, y, largerEq);
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm03(y, x, largerEq, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      return algorithm03(x, y, largerEq, false);
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, largerEq);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return largerEq(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return largerEq(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return largerEq(x, matrix(y));
    },
    'SparseMatrix, any': function SparseMatrixAny(x, y) {
      return algorithm12(x, y, largerEq, false);
    },
    'DenseMatrix, any': function DenseMatrixAny(x, y) {
      return algorithm14(x, y, largerEq, false);
    },
    'any, SparseMatrix': function anySparseMatrix(x, y) {
      return algorithm12(y, x, largerEq, true);
    },
    'any, DenseMatrix': function anyDenseMatrix(x, y) {
      return algorithm14(y, x, largerEq, true);
    },
    'Array, any': function ArrayAny(x, y) {
      // use matrix implementation
      return algorithm14(matrix(x), y, largerEq, false).valueOf();
    },
    'any, Array': function anyArray(x, y) {
      // use matrix implementation
      return algorithm14(matrix(y), x, largerEq, true).valueOf();
    }
  });
  return largerEq;
});
export var createLargerEqNumber = /* #__PURE__ */factory(name, ['typed', 'config'], function (_ref2) {
  var typed = _ref2.typed,
      config = _ref2.config;
  return typed(name, {
    'number, number': function numberNumber(x, y) {
      return x >= y || nearlyEqual(x, y, config.epsilon);
    }
  });
});