import { factory } from '../../utils/factory';
import { createAlgorithm01 } from '../../type/matrix/utils/algorithm01';
import { createAlgorithm02 } from '../../type/matrix/utils/algorithm02';
import { createAlgorithm06 } from '../../type/matrix/utils/algorithm06';
import { createAlgorithm11 } from '../../type/matrix/utils/algorithm11';
import { createAlgorithm13 } from '../../type/matrix/utils/algorithm13';
import { createAlgorithm14 } from '../../type/matrix/utils/algorithm14';
import { nthRootNumber } from '../../plain/number';
var name = 'nthRoot';
var dependencies = ['typed', 'matrix', 'equalScalar', 'BigNumber'];
export var createNthRoot = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var typed = _ref.typed,
      matrix = _ref.matrix,
      equalScalar = _ref.equalScalar,
      _BigNumber = _ref.BigNumber;
  var algorithm01 = createAlgorithm01({
    typed: typed
  });
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
   * Calculate the nth root of a value.
   * The principal nth root of a positive real number A, is the positive real
   * solution of the equation
   *
   *     x^root = A
   *
   * For matrices, the function is evaluated element wise.
   *
   * Syntax:
   *
   *     math.nthRoot(a)
   *     math.nthRoot(a, root)
   *
   * Examples:
   *
   *     math.nthRoot(9, 2)    // returns 3, as 3^2 == 9
   *     math.sqrt(9)          // returns 3, as 3^2 == 9
   *     math.nthRoot(64, 3)   // returns 4, as 4^3 == 64
   *
   * See also:
   *
   *     sqrt, pow
   *
   * @param {number | BigNumber | Array | Matrix | Complex} a
   *              Value for which to calculate the nth root
   * @param {number | BigNumber} [root=2]    The root.
   * @return {number | Complex | Array | Matrix} Returns the nth root of `a`
   */

  var complexErr = '' + 'Complex number not supported in function nthRoot. ' + 'Use nthRoots instead.';
  var nthRoot = typed(name, {
    number: function number(x) {
      return nthRootNumber(x, 2);
    },
    'number, number': nthRootNumber,
    BigNumber: function BigNumber(x) {
      return _bigNthRoot(x, new _BigNumber(2));
    },
    Complex: function Complex(x) {
      throw new Error(complexErr);
    },
    'Complex, number': function ComplexNumber(x, y) {
      throw new Error(complexErr);
    },
    'BigNumber, BigNumber': _bigNthRoot,
    'Array | Matrix': function ArrayMatrix(x) {
      return nthRoot(x, 2);
    },
    'SparseMatrix, SparseMatrix': function SparseMatrixSparseMatrix(x, y) {
      // density must be one (no zeros in matrix)
      if (y.density() === 1) {
        // sparse + sparse
        return algorithm06(x, y, nthRoot);
      } else {
        // throw exception
        throw new Error('Root must be non-zero');
      }
    },
    'SparseMatrix, DenseMatrix': function SparseMatrixDenseMatrix(x, y) {
      return algorithm02(y, x, nthRoot, true);
    },
    'DenseMatrix, SparseMatrix': function DenseMatrixSparseMatrix(x, y) {
      // density must be one (no zeros in matrix)
      if (y.density() === 1) {
        // dense + sparse
        return algorithm01(x, y, nthRoot, false);
      } else {
        // throw exception
        throw new Error('Root must be non-zero');
      }
    },
    'DenseMatrix, DenseMatrix': function DenseMatrixDenseMatrix(x, y) {
      return algorithm13(x, y, nthRoot);
    },
    'Array, Array': function ArrayArray(x, y) {
      // use matrix implementation
      return nthRoot(matrix(x), matrix(y)).valueOf();
    },
    'Array, Matrix': function ArrayMatrix(x, y) {
      // use matrix implementation
      return nthRoot(matrix(x), y);
    },
    'Matrix, Array': function MatrixArray(x, y) {
      // use matrix implementation
      return nthRoot(x, matrix(y));
    },
    'SparseMatrix, number | BigNumber': function SparseMatrixNumberBigNumber(x, y) {
      return algorithm11(x, y, nthRoot, false);
    },
    'DenseMatrix, number | BigNumber': function DenseMatrixNumberBigNumber(x, y) {
      return algorithm14(x, y, nthRoot, false);
    },
    'number | BigNumber, SparseMatrix': function numberBigNumberSparseMatrix(x, y) {
      // density must be one (no zeros in matrix)
      if (y.density() === 1) {
        // sparse - scalar
        return algorithm11(y, x, nthRoot, true);
      } else {
        // throw exception
        throw new Error('Root must be non-zero');
      }
    },
    'number | BigNumber, DenseMatrix': function numberBigNumberDenseMatrix(x, y) {
      return algorithm14(y, x, nthRoot, true);
    },
    'Array, number | BigNumber': function ArrayNumberBigNumber(x, y) {
      // use matrix implementation
      return nthRoot(matrix(x), y).valueOf();
    },
    'number | BigNumber, Array': function numberBigNumberArray(x, y) {
      // use matrix implementation
      return nthRoot(x, matrix(y)).valueOf();
    }
  });
  return nthRoot;
  /**
   * Calculate the nth root of a for BigNumbers, solve x^root == a
   * https://rosettacode.org/wiki/Nth_root#JavaScript
   * @param {BigNumber} a
   * @param {BigNumber} root
   * @private
   */

  function _bigNthRoot(a, root) {
    var precision = _BigNumber.precision;

    var Big = _BigNumber.clone({
      precision: precision + 2
    });

    var zero = new _BigNumber(0);
    var one = new Big(1);
    var inv = root.isNegative();

    if (inv) {
      root = root.neg();
    }

    if (root.isZero()) {
      throw new Error('Root must be non-zero');
    }

    if (a.isNegative() && !root.abs().mod(2).equals(1)) {
      throw new Error('Root must be odd when a is negative.');
    } // edge cases zero and infinity


    if (a.isZero()) {
      return inv ? new Big(Infinity) : 0;
    }

    if (!a.isFinite()) {
      return inv ? zero : a;
    }

    var x = a.abs().pow(one.div(root)); // If a < 0, we require that root is an odd integer,
    // so (-1) ^ (1/root) = -1

    x = a.isNeg() ? x.neg() : x;
    return new _BigNumber((inv ? one.div(x) : x).toPrecision(precision));
  }
});
export var createNthRootNumber = /* #__PURE__ */factory(name, ['typed'], function (_ref2) {
  var typed = _ref2.typed;
  return typed(name, {
    number: nthRootNumber,
    'number, number': nthRootNumber
  });
});