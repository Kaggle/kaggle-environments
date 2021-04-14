import { factory } from '../../utils/factory';
import { createAlgorithm03 } from '../../type/matrix/utils/algorithm03';
import { createAlgorithm07 } from '../../type/matrix/utils/algorithm07';
import { createAlgorithm11 } from '../../type/matrix/utils/algorithm11';
import { createAlgorithm12 } from '../../type/matrix/utils/algorithm12';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
var name = 'dotPow';
var dependencies = ['typed', 'equalScalar', 'matrix', 'pow', 'DenseMatrix'];
export var createDotPow = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      equalScalar = _ref.equalScalar,
      matrix = _ref.matrix,
      pow = _ref.pow,
      DenseMatrix = _ref.DenseMatrix;
  var algorithm03 = createAlgorithm03({
    typed: typed
  });
  var algorithm07 = createAlgorithm07({
    typed: typed,
    DenseMatrix: DenseMatrix
  });
  var algorithm11 = createAlgorithm11({
    typed: typed,
    equalScalar: equalScalar
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
   * Calculates the power of x to y element wise.
   *
   * Syntax:
   *
   *    math.dotPow(x, y)
   *
   * Examples:
   *
   *    math.dotPow(2, 3)            // returns number 8
   *
   *    const a = [[1, 2], [4, 3]]
   *    math.dotPow(a, 2)            // returns Array [[1, 4], [16, 9]]
   *    math.pow(a, 2)               // returns Array [[9, 8], [16, 17]]
   *
   * See also:
   *
   *    pow, sqrt, multiply
   *
   * @param  {number | BigNumber | Complex | Unit | Array | Matrix} x  The base
   * @param  {number | BigNumber | Complex | Unit | Array | Matrix} y  The exponent
   * @return {number | BigNumber | Complex | Unit | Array | Matrix}                     The value of `x` to the power `y`
   */

  var dotPow = typed(name, {
    'any, any': pow,
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      return algorithm07(x, y, pow, false);
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm03(y, x, pow, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      return algorithm03(x, y, pow, false);
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, pow);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return dotPow(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return dotPow(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return dotPow(x, matrix(y));
    },
    'SparseMatrix, any': function SparseMatrixAny(x, y) {
      return algorithm11(x, y, dotPow, false);
    },
    'DenseMatrix, any': function DenseMatrixAny(x, y) {
      return algorithm14(x, y, dotPow, false);
    },
    'any, SparseMatrix': function anySparseMatrix(x, y) {
      return algorithm12(y, x, dotPow, true);
    },
    'any, DenseMatrix': function anyDenseMatrix(x, y) {
      return algorithm14(y, x, dotPow, true);
    },
    'Array, any': function ArrayAny(x, y) {
      // use matrix implementation
      return algorithm14(matrix(x), y, dotPow, false).valueOf();
    },
    'any, Array': function anyArray(x, y) {
      // use matrix implementation
      return algorithm14(matrix(y), x, dotPow, true).valueOf();
    }
  });
  return dotPow;
});