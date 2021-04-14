import { nearlyEqual as bigNearlyEqual } from '../../utils/bignumber/nearlyEqual';
import { nearlyEqual } from '../../utils/number';
import { factory } from '../../utils/factory';
import { createAlgorithm03 } from '../../type/matrix/utils/algorithm03';
import { createAlgorithm07 } from '../../type/matrix/utils/algorithm07';
import { createAlgorithm12 } from '../../type/matrix/utils/algorithm12';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
var name = 'smallerEq';
var dependencies = ['typed', 'config', 'matrix', 'DenseMatrix'];
export var createSmallerEq = /* #__PURE__ */factory(name, dependencies, function (_ref) {
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
   * Test whether value x is smaller or equal to y.
   *
   * The function returns true when x is smaller than y or the relative
   * difference between x and y is smaller than the configured epsilon. The
   * function cannot be used to compare values smaller than approximately 2.22e-16.
   *
   * For matrices, the function is evaluated element wise.
   * Strings are compared by their numerical value.
   *
   * Syntax:
   *
   *    math.smallerEq(x, y)
   *
   * Examples:
   *
   *    math.smaller(1 + 2, 3)        // returns false
   *    math.smallerEq(1 + 2, 3)      // returns true
   *
   * See also:
   *
   *    equal, unequal, smaller, larger, largerEq, compare
   *
   * @param  {number | BigNumber | Fraction | boolean | Unit | string | Array | Matrix} x First value to compare
   * @param  {number | BigNumber | Fraction | boolean | Unit | string | Array | Matrix} y Second value to compare
   * @return {boolean | Array | Matrix} Returns true when the x is smaller than y, else returns false
   */

  var smallerEq = typed(name, {
    'boolean, boolean': function booleanBoolean(x, y) {
      return x <= y;
    },
    'number, number': function numberNumber(x, y) {
      return x <= y || nearlyEqual(x, y, config.epsilon);
    },
    'BigNumber, BigNumber': function BigNumberBigNumber(x, y) {
      return x.lte(y) || bigNearlyEqual(x, y, config.epsilon);
    },
    'Fraction, Fraction': function FractionFraction(x, y) {
      return x.compare(y) !== 1;
    },
    'Complex, Complex': function ComplexComplex() {
      throw new TypeError('No ordering relation is defined for complex numbers');
    },
    'Unit, Unit': function UnitUnit(x, y) {
      if (!x.equalBase(y)) {
        throw new Error('Cannot compare units with different base');
      }

      return smallerEq(x.value, y.value);
    },
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      return algorithm07(x, y, smallerEq);
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm03(y, x, smallerEq, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      return algorithm03(x, y, smallerEq, false);
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, smallerEq);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return smallerEq(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return smallerEq(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return smallerEq(x, matrix(y));
    },
    'SparseMatrix, any': function SparseMatrixAny(x, y) {
      return algorithm12(x, y, smallerEq, false);
    },
    'DenseMatrix, any': function DenseMatrixAny(x, y) {
      return algorithm14(x, y, smallerEq, false);
    },
    'any, SparseMatrix': function anySparseMatrix(x, y) {
      return algorithm12(y, x, smallerEq, true);
    },
    'any, DenseMatrix': function anyDenseMatrix(x, y) {
      return algorithm14(y, x, smallerEq, true);
    },
    'Array, any': function ArrayAny(x, y) {
      // use matrix implementation
      return algorithm14(matrix(x), y, smallerEq, false).valueOf();
    },
    'any, Array': function anyArray(x, y) {
      // use matrix implementation
      return algorithm14(matrix(y), x, smallerEq, true).valueOf();
    }
  });
  return smallerEq;
});
export var createSmallerEqNumber = /* #__PURE__ */factory(name, ['typed', 'config'], function (_ref2) {
  var typed = _ref2.typed,
      config = _ref2.config;
  return typed(name, {
    'number, number': function numberNumber(x, y) {
      return x <= y || nearlyEqual(x, y, config.epsilon);
    }
  });
});