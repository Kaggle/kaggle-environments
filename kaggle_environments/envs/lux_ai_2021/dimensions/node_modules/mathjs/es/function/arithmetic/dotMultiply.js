import { factory } from '../../utils/factory';
import { createAlgorithm02 } from '../../type/matrix/utils/algorithm02';
import { createAlgorithm09 } from '../../type/matrix/utils/algorithm09';
import { createAlgorithm11 } from '../../type/matrix/utils/algorithm11';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
var name = 'dotMultiply';
var dependencies = ['typed', 'matrix', 'equalScalar', 'multiplyScalar'];
export var createDotMultiply = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix,
      equalScalar = _ref.equalScalar,
      multiplyScalar = _ref.multiplyScalar;
  var algorithm02 = createAlgorithm02({
    typed: typed,
    equalScalar: equalScalar
  });
  var algorithm09 = createAlgorithm09({
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
   * Multiply two matrices element wise. The function accepts both matrices and
   * scalar values.
   *
   * Syntax:
   *
   *    math.dotMultiply(x, y)
   *
   * Examples:
   *
   *    math.dotMultiply(2, 4) // returns 8
   *
   *    a = [[9, 5], [6, 1]]
   *    b = [[3, 2], [5, 2]]
   *
   *    math.dotMultiply(a, b) // returns [[27, 10], [30, 2]]
   *    math.multiply(a, b)    // returns [[52, 28], [23, 14]]
   *
   * See also:
   *
   *    multiply, divide, dotDivide
   *
   * @param  {number | BigNumber | Fraction | Complex | Unit | Array | Matrix} x Left hand value
   * @param  {number | BigNumber | Fraction | Complex | Unit | Array | Matrix} y Right hand value
   * @return {number | BigNumber | Fraction | Complex | Unit | Array | Matrix}                    Multiplication of `x` and `y`
   */

  var dotMultiply = typed(name, {
    'any, any': multiplyScalar,
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      return algorithm09(x, y, multiplyScalar, false);
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm02(y, x, multiplyScalar, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      return algorithm02(x, y, multiplyScalar, false);
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, multiplyScalar);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return dotMultiply(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return dotMultiply(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return dotMultiply(x, matrix(y));
    },
    'SparseMatrix, any': function SparseMatrixAny(x, y) {
      return algorithm11(x, y, multiplyScalar, false);
    },
    'DenseMatrix, any': function DenseMatrixAny(x, y) {
      return algorithm14(x, y, multiplyScalar, false);
    },
    'any, SparseMatrix': function anySparseMatrix(x, y) {
      return algorithm11(y, x, multiplyScalar, true);
    },
    'any, DenseMatrix': function anyDenseMatrix(x, y) {
      return algorithm14(y, x, multiplyScalar, true);
    },
    'Array, any': function ArrayAny(x, y) {
      // use matrix implementation
      return algorithm14(matrix(x), y, multiplyScalar, false).valueOf();
    },
    'any, Array': function anyArray(x, y) {
      // use matrix implementation
      return algorithm14(matrix(y), x, multiplyScalar, true).valueOf();
    }
  });
  return dotMultiply;
});