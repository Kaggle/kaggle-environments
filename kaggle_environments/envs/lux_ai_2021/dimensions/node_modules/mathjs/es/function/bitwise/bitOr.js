import { bitOrBigNumber } from '../../utils/bignumber/bitwise';
import { factory } from '../../utils/factory';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
import { createAlgorithm10 } from '../../type/matrix/utils/algorithm10';
import { createAlgorithm04 } from '../../type/matrix/utils/algorithm04';
import { createAlgorithm01 } from '../../type/matrix/utils/algorithm01';
import { bitOrNumber } from '../../plain/number';
var name = 'bitOr';
var dependencies = ['typed', 'matrix', 'equalScalar', 'DenseMatrix'];
export var createBitOr = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix,
      equalScalar = _ref.equalScalar,
      DenseMatrix = _ref.DenseMatrix;
  var algorithm01 = createAlgorithm01({
    typed: typed
  });
  var algorithm04 = createAlgorithm04({
    typed: typed,
    equalScalar: equalScalar
  });
  var algorithm10 = createAlgorithm10({
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
   * Bitwise OR two values, `x | y`.
   * For matrices, the function is evaluated element wise.
   * For units, the function is evaluated on the lowest print base.
   *
   * Syntax:
   *
   *    math.bitOr(x, y)
   *
   * Examples:
   *
   *    math.bitOr(1, 2)               // returns number 3
   *
   *    math.bitOr([1, 2, 3], 4)       // returns Array [5, 6, 7]
   *
   * See also:
   *
   *    bitAnd, bitNot, bitXor, leftShift, rightArithShift, rightLogShift
   *
   * @param  {number | BigNumber | Array | Matrix} x First value to or
   * @param  {number | BigNumber | Array | Matrix} y Second value to or
   * @return {number | BigNumber | Array | Matrix} OR of `x` and `y`
   */

  var bitOr = typed(name, {
    'number, number': bitOrNumber,
    'BigNumber, BigNumber': bitOrBigNumber,
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      return algorithm04(x, y, bitOr);
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm01(y, x, bitOr, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      return algorithm01(x, y, bitOr, false);
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, bitOr);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return bitOr(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return bitOr(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return bitOr(x, matrix(y));
    },
    'SparseMatrix, any': function SparseMatrixAny(x, y) {
      return algorithm10(x, y, bitOr, false);
    },
    'DenseMatrix, any': function DenseMatrixAny(x, y) {
      return algorithm14(x, y, bitOr, false);
    },
    'any, SparseMatrix': function anySparseMatrix(x, y) {
      return algorithm10(y, x, bitOr, true);
    },
    'any, DenseMatrix': function anyDenseMatrix(x, y) {
      return algorithm14(y, x, bitOr, true);
    },
    'Array, any': function ArrayAny(x, y) {
      // use matrix implementation
      return algorithm14(matrix(x), y, bitOr, false).valueOf();
    },
    'any, Array': function anyArray(x, y) {
      // use matrix implementation
      return algorithm14(matrix(y), x, bitOr, true).valueOf();
    }
  });
  return bitOr;
});