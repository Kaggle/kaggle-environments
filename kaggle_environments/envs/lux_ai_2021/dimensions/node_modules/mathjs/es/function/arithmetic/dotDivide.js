import { factory } from '../../utils/factory';
import { createAlgorithm02 } from '../../type/matrix/utils/algorithm02';
import { createAlgorithm03 } from '../../type/matrix/utils/algorithm03';
import { createAlgorithm07 } from '../../type/matrix/utils/algorithm07';
import { createAlgorithm11 } from '../../type/matrix/utils/algorithm11';
import { createAlgorithm12 } from '../../type/matrix/utils/algorithm12';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
var name = 'dotDivide';
var dependencies = ['typed', 'matrix', 'equalScalar', 'divideScalar', 'DenseMatrix'];
export var createDotDivide = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix,
      equalScalar = _ref.equalScalar,
      divideScalar = _ref.divideScalar,
      DenseMatrix = _ref.DenseMatrix;
  var algorithm02 = createAlgorithm02({
    typed: typed,
    equalScalar: equalScalar
  });
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
   * Divide two matrices element wise. The function accepts both matrices and
   * scalar values.
   *
   * Syntax:
   *
   *    math.dotDivide(x, y)
   *
   * Examples:
   *
   *    math.dotDivide(2, 4)   // returns 0.5
   *
   *    a = [[9, 5], [6, 1]]
   *    b = [[3, 2], [5, 2]]
   *
   *    math.dotDivide(a, b)   // returns [[3, 2.5], [1.2, 0.5]]
   *    math.divide(a, b)      // returns [[1.75, 0.75], [-1.75, 2.25]]
   *
   * See also:
   *
   *    divide, multiply, dotMultiply
   *
   * @param  {number | BigNumber | Fraction | Complex | Unit | Array | Matrix} x Numerator
   * @param  {number | BigNumber | Fraction | Complex | Unit | Array | Matrix} y Denominator
   * @return {number | BigNumber | Fraction | Complex | Unit | Array | Matrix}                    Quotient, `x ./ y`
   */

  var dotDivide = typed(name, {
    'any, any': divideScalar,
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      return algorithm07(x, y, divideScalar, false);
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm02(y, x, divideScalar, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      return algorithm03(x, y, divideScalar, false);
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, divideScalar);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return dotDivide(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return dotDivide(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return dotDivide(x, matrix(y));
    },
    'SparseMatrix, any': function SparseMatrixAny(x, y) {
      return algorithm11(x, y, divideScalar, false);
    },
    'DenseMatrix, any': function DenseMatrixAny(x, y) {
      return algorithm14(x, y, divideScalar, false);
    },
    'any, SparseMatrix': function anySparseMatrix(x, y) {
      return algorithm12(y, x, divideScalar, true);
    },
    'any, DenseMatrix': function anyDenseMatrix(x, y) {
      return algorithm14(y, x, divideScalar, true);
    },
    'Array, any': function ArrayAny(x, y) {
      // use matrix implementation
      return algorithm14(matrix(x), y, divideScalar, false).valueOf();
    },
    'any, Array': function anyArray(x, y) {
      // use matrix implementation
      return algorithm14(matrix(y), x, divideScalar, true).valueOf();
    }
  });
  return dotDivide;
});