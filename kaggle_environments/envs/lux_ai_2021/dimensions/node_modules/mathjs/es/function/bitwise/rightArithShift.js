import { rightArithShiftBigNumber } from '../../utils/bignumber/bitwise';
import { createAlgorithm02 } from '../../type/matrix/utils/algorithm02';
import { createAlgorithm11 } from '../../type/matrix/utils/algorithm11';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
import { createAlgorithm01 } from '../../type/matrix/utils/algorithm01';
import { createAlgorithm10 } from '../../type/matrix/utils/algorithm10';
import { createAlgorithm08 } from '../../type/matrix/utils/algorithm08';
import { factory } from '../../utils/factory';
import { rightArithShiftNumber } from '../../plain/number';
var name = 'rightArithShift';
var dependencies = ['typed', 'matrix', 'equalScalar', 'zeros', 'DenseMatrix'];
export var createRightArithShift = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix,
      equalScalar = _ref.equalScalar,
      zeros = _ref.zeros,
      DenseMatrix = _ref.DenseMatrix;
  var algorithm01 = createAlgorithm01({
    typed: typed
  });
  var algorithm02 = createAlgorithm02({
    typed: typed,
    equalScalar: equalScalar
  });
  var algorithm08 = createAlgorithm08({
    typed: typed,
    equalScalar: equalScalar
  });
  var algorithm10 = createAlgorithm10({
    typed: typed,
    DenseMatrix: DenseMatrix
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
   * Bitwise right arithmetic shift of a value x by y number of bits, `x >> y`.
   * For matrices, the function is evaluated element wise.
   * For units, the function is evaluated on the best prefix base.
   *
   * Syntax:
   *
   *    math.rightArithShift(x, y)
   *
   * Examples:
   *
   *    math.rightArithShift(4, 2)               // returns number 1
   *
   *    math.rightArithShift([16, -32, 64], 4)   // returns Array [1, -2, 3]
   *
   * See also:
   *
   *    bitAnd, bitNot, bitOr, bitXor, rightArithShift, rightLogShift
   *
   * @param  {number | BigNumber | Array | Matrix} x Value to be shifted
   * @param  {number | BigNumber} y Amount of shifts
   * @return {number | BigNumber | Array | Matrix} `x` sign-filled shifted right `y` times
   */

  var rightArithShift = typed(name, {
    'number, number': rightArithShiftNumber,
    'BigNumber, BigNumber': rightArithShiftBigNumber,
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      return algorithm08(x, y, rightArithShift, false);
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm02(y, x, rightArithShift, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      return algorithm01(x, y, rightArithShift, false);
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, rightArithShift);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return rightArithShift(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return rightArithShift(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return rightArithShift(x, matrix(y));
    },
    'SparseMatrix, number | BigNumber': function SparseMatrixNumberBigNumber(x, y) {
      // check scalar
      if (equalScalar(y, 0)) {
        return x.clone();
      }

      return algorithm11(x, y, rightArithShift, false);
    },
    'DenseMatrix, number | BigNumber': function DenseMatrixNumberBigNumber(x, y) {
      // check scalar
      if (equalScalar(y, 0)) {
        return x.clone();
      }

      return algorithm14(x, y, rightArithShift, false);
    },
    'number | BigNumber, SparseMatrix': function numberBigNumberSparseMatrix(x, y) {
      // check scalar
      if (equalScalar(x, 0)) {
        return zeros(y.size(), y.storage());
      }

      return algorithm10(y, x, rightArithShift, true);
    },
    'number | BigNumber, DenseMatrix': function numberBigNumberDenseMatrix(x, y) {
      // check scalar
      if (equalScalar(x, 0)) {
        return zeros(y.size(), y.storage());
      }

      return algorithm14(y, x, rightArithShift, true);
    },
    'Array, number | BigNumber': function ArrayNumberBigNumber(x, y) {
      // use matrix implementation
      return rightArithShift(matrix(x), y).valueOf();
    },
    'number | BigNumber, Array': function numberBigNumberArray(x, y) {
      // use matrix implementation
      return rightArithShift(x, matrix(y)).valueOf();
    }
  });
  return rightArithShift;
});