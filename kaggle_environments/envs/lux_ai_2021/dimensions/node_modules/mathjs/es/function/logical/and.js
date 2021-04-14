import { createAlgorithm02 } from '../../type/matrix/utils/algorithm02';
import { createAlgorithm11 } from '../../type/matrix/utils/algorithm11';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
import { createAlgorithm06 } from '../../type/matrix/utils/algorithm06';
import { factory } from '../../utils/factory';
import { andNumber } from '../../plain/number';
var name = 'and';
var dependencies = ['typed', 'matrix', 'equalScalar', 'zeros', 'not'];
export var createAnd = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix,
      equalScalar = _ref.equalScalar,
      zeros = _ref.zeros,
      not = _ref.not;
  var algorithm02 = createAlgorithm02({
    typed: typed,
    equalScalar: equalScalar
  });
  var algorithm06 = createAlgorithm06({
    typed: typed,
    equalScalar: equalScalar
  });
  var algorithm11 = createAlgorithm11({
    typed: typed,
    equalScalar: equalScalar
  });
  var algorithm13 = createAlgorithm13({
    typed: typed
  });
  var algorithm14 = createAlgorithm14({
    typed: typed
  });
  /**
   * Logical `and`. Test whether two values are both defined with a nonzero/nonempty value.
   * For matrices, the function is evaluated element wise.
   *
   * Syntax:
   *
   *    math.and(x, y)
   *
   * Examples:
   *
   *    math.and(2, 4)   // returns true
   *
   *    a = [2, 0, 0]
   *    b = [3, 7, 0]
   *    c = 0
   *
   *    math.and(a, b)   // returns [true, false, false]
   *    math.and(a, c)   // returns [false, false, false]
   *
   * See also:
   *
   *    not, or, xor
   *
   * @param  {number | BigNumber | Complex | Unit | Array | Matrix} x First value to check
   * @param  {number | BigNumber | Complex | Unit | Array | Matrix} y Second value to check
   * @return {boolean | Array | Matrix}
   *            Returns true when both inputs are defined with a nonzero/nonempty value.
   */

  var and = typed(name, {
    'number, number': andNumber,
    'Complex, Complex': function ComplexComplex(x, y) {
      return (x.re !== 0 || x.im !== 0) && (y.re !== 0 || y.im !== 0);
    },
    'BigNumber, BigNumber': function BigNumberBigNumber(x, y) {
      return !x.isZero() && !y.isZero() && !x.isNaN() && !y.isNaN();
    },
    'Unit, Unit': function UnitUnit(x, y) {
      return and(x.value || 0, y.value || 0);
    },
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      return algorithm06(x, y, and, false);
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm02(y, x, and, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      return algorithm02(x, y, and, false);
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, and);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return and(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return and(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return and(x, matrix(y));
    },
    'SparseMatrix, any': function SparseMatrixAny(x, y) {
      // check scalar
      if (not(y)) {
        // return zero matrix
        return zeros(x.size(), x.storage());
      }

      return algorithm11(x, y, and, false);
    },
    'DenseMatrix, any': function DenseMatrixAny(x, y) {
      // check scalar
      if (not(y)) {
        // return zero matrix
        return zeros(x.size(), x.storage());
      }

      return algorithm14(x, y, and, false);
    },
    'any, SparseMatrix': function anySparseMatrix(x, y) {
      // check scalar
      if (not(x)) {
        // return zero matrix
        return zeros(x.size(), x.storage());
      }

      return algorithm11(y, x, and, true);
    },
    'any, DenseMatrix': function anyDenseMatrix(x, y) {
      // check scalar
      if (not(x)) {
        // return zero matrix
        return zeros(x.size(), x.storage());
      }

      return algorithm14(y, x, and, true);
    },
    'Array, any': function ArrayAny(x, y) {
      // use matrix implementation
      return and(matrix(x), y).valueOf();
    },
    'any, Array': function anyArray(x, y) {
      // use matrix implementation
      return and(x, matrix(y)).valueOf();
    }
  });
  return and;
});