import { clone } from '../../utils/object';
import { factory } from '../../utils/factory';
import { format } from '../../utils/string';
var name = 'eigs';
var dependencies = ['config', 'typed', 'matrix', 'addScalar', 'equal', 'subtract', 'abs', 'atan', 'cos', 'sin', 'multiplyScalar', 'inv', 'bignumber', 'multiply', 'add'];
export var createEigs = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var config = _ref.config,
      typed = _ref.typed,
      matrix = _ref.matrix,
      addScalar = _ref.addScalar,
      subtract = _ref.subtract,
      equal = _ref.equal,
      abs = _ref.abs,
      atan = _ref.atan,
      cos = _ref.cos,
      sin = _ref.sin,
      multiplyScalar = _ref.multiplyScalar,
      inv = _ref.inv,
      bignumber = _ref.bignumber,
      multiply = _ref.multiply,
      add = _ref.add;

  /**
   * Compute eigenvalue and eigenvector of a real symmetric matrix.
   * Only applicable to two dimensional symmetric matrices. Uses Jacobi
   * Algorithm. Matrix containing mixed type ('number', 'bignumber', 'fraction')
   * of elements are not supported. Input matrix or 2D array should contain all elements
   * of either 'number', 'bignumber' or 'fraction' type. For 'number' and 'fraction', the
   * eigenvalues are of 'number' type. For 'bignumber' the eigenvalues are of ''bignumber' type.
   * Eigenvectors are always of 'number' type.
   *
   * Syntax:
   *
   *     math.eigs(x)
   *
   * Examples:
   *
   *     const H = [[5, 2.3], [2.3, 1]]
   *     const ans = math.eigs(H) // returns {values: [E1,E2...sorted], vectors: [v1,v2.... corresponding vectors]}
   *     const E = ans.values
   *     const U = ans.vectors
   *     const UTxHxU = math.multiply(math.transpose(U), H, U) // rotates H to the eigen-representation
   *     E[0] == UTxHxU[0][0]  // returns true
   * See also:
   *
   *     inv
   *
   * @param {Array | Matrix} x  Matrix to be diagonalized
   * @return {{values: Array, vectors: Array} | {values: Matrix, vectors: Matrix}} Object containing eigenvalues (Array or Matrix) and eigenvectors (2D Array/Matrix).
   */
  var eigs = typed('eigs', {
    Array: function Array(x) {
      // check array size
      var mat = matrix(x);
      var size = mat.size();

      if (size.length !== 2 || size[0] !== size[1]) {
        throw new RangeError('Matrix must be square ' + '(size: ' + format(size) + ')');
      } // use dense 2D matrix implementation


      var ans = checkAndSubmit(mat, size[0]);
      return {
        values: ans[0],
        vectors: ans[1]
      };
    },
    Matrix: function Matrix(x) {
      // use dense 2D array implementation
      // dense matrix
      var size = x.size();

      if (size.length !== 2 || size[0] !== size[1]) {
        throw new RangeError('Matrix must be square ' + '(size: ' + format(size) + ')');
      }

      var ans = checkAndSubmit(x, size[0]);
      return {
        values: matrix(ans[0]),
        vectors: matrix(ans[1])
      };
    }
  }); // Is the matrix
  // symmetric ?

  function isSymmetric(x, n) {
    for (var i = 0; i < n; i++) {
      for (var j = i; j < n; j++) {
        // not symmtric
        if (!equal(x[i][j], x[j][i])) {
          throw new TypeError('Input matrix is not symmetric');
        }
      }
    }
  } // check input for possible problems
  // and perform diagonalization efficiently for
  // specific type of number


  function checkAndSubmit(x, n) {
    var type = x.datatype(); // type check

    if (type === undefined) {
      type = x.getDataType();
    }

    if (type !== 'number' && type !== 'BigNumber' && type !== 'Fraction') {
      if (type === 'mixed') {
        throw new TypeError('Mixed matrix element type is not supported');
      } else {
        throw new TypeError('Matrix element type not supported (' + type + ')');
      }
    } else {
      isSymmetric(x.toArray(), n);
    } // perform efficient calculation for 'numbers'


    if (type === 'number') {
      return diag(x.toArray());
    } else if (type === 'Fraction') {
      var xArr = x.toArray(); // convert fraction to numbers

      for (var i = 0; i < n; i++) {
        for (var j = i; j < n; j++) {
          xArr[i][j] = xArr[i][j].valueOf();
          xArr[j][i] = xArr[i][j];
        }
      }

      return diag(x.toArray());
    } else if (type === 'BigNumber') {
      return diagBig(x.toArray());
    }
  } // diagonalization implementation for number (efficient)


  function diag(x) {
    var N = x.length;
    var e0 = Math.abs(config.epsilon / N);
    var psi;
    var Sij = new Array(N); // Sij is Identity Matrix

    for (var i = 0; i < N; i++) {
      Sij[i] = createArray(N, 0);
      Sij[i][i] = 1.0;
    } // initial error


    var Vab = getAij(x);

    while (Math.abs(Vab[1]) >= Math.abs(e0)) {
      var _i = Vab[0][0];
      var j = Vab[0][1];
      psi = getTheta(x[_i][_i], x[j][j], x[_i][j]);
      x = x1(x, psi, _i, j);
      Sij = Sij1(Sij, psi, _i, j);
      Vab = getAij(x);
    }

    var Ei = createArray(N, 0); // eigenvalues

    for (var _i2 = 0; _i2 < N; _i2++) {
      Ei[_i2] = x[_i2][_i2];
    }

    return sorting(clone(Ei), clone(Sij));
  } // diagonalization implementation for bigNumber


  function diagBig(x) {
    var N = x.length;
    var e0 = abs(config.epsilon / N);
    var psi;
    var Sij = new Array(N); // Sij is Identity Matrix

    for (var i = 0; i < N; i++) {
      Sij[i] = createArray(N, 0);
      Sij[i][i] = 1.0;
    } // initial error


    var Vab = getAijBig(x);

    while (abs(Vab[1]) >= abs(e0)) {
      var _i3 = Vab[0][0];
      var j = Vab[0][1];
      psi = getThetaBig(x[_i3][_i3], x[j][j], x[_i3][j]);
      x = x1Big(x, psi, _i3, j);
      Sij = Sij1Big(Sij, psi, _i3, j);
      Vab = getAijBig(x);
    }

    var Ei = createArray(N, 0); // eigenvalues

    for (var _i4 = 0; _i4 < N; _i4++) {
      Ei[_i4] = x[_i4][_i4];
    } // return [clone(Ei), clone(Sij)]


    return sorting(clone(Ei), clone(Sij));
  } // get angle


  function getTheta(aii, ajj, aij) {
    var denom = ajj - aii;

    if (Math.abs(denom) <= config.epsilon) {
      return Math.PI / 4;
    } else {
      return 0.5 * Math.atan(2 * aij / (ajj - aii));
    }
  } // get angle


  function getThetaBig(aii, ajj, aij) {
    var denom = subtract(ajj, aii);

    if (abs(denom) <= config.epsilon) {
      return bignumber(-1).acos().div(4);
    } else {
      return multiplyScalar(0.5, atan(multiply(2, aij, inv(denom))));
    }
  } // update eigvec


  function Sij1(Sij, theta, i, j) {
    var N = Sij.length;
    var c = Math.cos(theta);
    var s = Math.sin(theta);
    var Ski = createArray(N, 0);
    var Skj = createArray(N, 0);

    for (var k = 0; k < N; k++) {
      Ski[k] = c * Sij[k][i] - s * Sij[k][j];
      Skj[k] = s * Sij[k][i] + c * Sij[k][j];
    }

    for (var _k = 0; _k < N; _k++) {
      Sij[_k][i] = Ski[_k];
      Sij[_k][j] = Skj[_k];
    }

    return Sij;
  } // update eigvec for overlap


  function Sij1Big(Sij, theta, i, j) {
    var N = Sij.length;
    var c = cos(theta);
    var s = sin(theta);
    var Ski = createArray(N, bignumber(0));
    var Skj = createArray(N, bignumber(0));

    for (var k = 0; k < N; k++) {
      Ski[k] = subtract(multiplyScalar(c, Sij[k][i]), multiplyScalar(s, Sij[k][j]));
      Skj[k] = addScalar(multiplyScalar(s, Sij[k][i]), multiplyScalar(c, Sij[k][j]));
    }

    for (var _k2 = 0; _k2 < N; _k2++) {
      Sij[_k2][i] = Ski[_k2];
      Sij[_k2][j] = Skj[_k2];
    }

    return Sij;
  } // update matrix


  function x1Big(Hij, theta, i, j) {
    var N = Hij.length;
    var c = bignumber(cos(theta));
    var s = bignumber(sin(theta));
    var c2 = multiplyScalar(c, c);
    var s2 = multiplyScalar(s, s);
    var Aki = createArray(N, bignumber(0));
    var Akj = createArray(N, bignumber(0)); // 2cs Hij

    var csHij = multiply(bignumber(2), c, s, Hij[i][j]); //  Aii

    var Aii = addScalar(subtract(multiplyScalar(c2, Hij[i][i]), csHij), multiplyScalar(s2, Hij[j][j]));
    var Ajj = add(multiplyScalar(s2, Hij[i][i]), csHij, multiplyScalar(c2, Hij[j][j])); // 0  to i

    for (var k = 0; k < N; k++) {
      Aki[k] = subtract(multiplyScalar(c, Hij[i][k]), multiplyScalar(s, Hij[j][k]));
      Akj[k] = addScalar(multiplyScalar(s, Hij[i][k]), multiplyScalar(c, Hij[j][k]));
    } // Modify Hij


    Hij[i][i] = Aii;
    Hij[j][j] = Ajj;
    Hij[i][j] = bignumber(0);
    Hij[j][i] = bignumber(0); // 0  to i

    for (var _k3 = 0; _k3 < N; _k3++) {
      if (_k3 !== i && _k3 !== j) {
        Hij[i][_k3] = Aki[_k3];
        Hij[_k3][i] = Aki[_k3];
        Hij[j][_k3] = Akj[_k3];
        Hij[_k3][j] = Akj[_k3];
      }
    }

    return Hij;
  } // update matrix


  function x1(Hij, theta, i, j) {
    var N = Hij.length;
    var c = Math.cos(theta);
    var s = Math.sin(theta);
    var c2 = c * c;
    var s2 = s * s;
    var Aki = createArray(N, 0);
    var Akj = createArray(N, 0); //  Aii

    var Aii = c2 * Hij[i][i] - 2 * c * s * Hij[i][j] + s2 * Hij[j][j];
    var Ajj = s2 * Hij[i][i] + 2 * c * s * Hij[i][j] + c2 * Hij[j][j]; // 0  to i

    for (var k = 0; k < N; k++) {
      Aki[k] = c * Hij[i][k] - s * Hij[j][k];
      Akj[k] = s * Hij[i][k] + c * Hij[j][k];
    } // Modify Hij


    Hij[i][i] = Aii;
    Hij[j][j] = Ajj;
    Hij[i][j] = 0;
    Hij[j][i] = 0; // 0  to i

    for (var _k4 = 0; _k4 < N; _k4++) {
      if (_k4 !== i && _k4 !== j) {
        Hij[i][_k4] = Aki[_k4];
        Hij[_k4][i] = Aki[_k4];
        Hij[j][_k4] = Akj[_k4];
        Hij[_k4][j] = Akj[_k4];
      }
    }

    return Hij;
  } // get max off-diagonal value from Upper Diagonal


  function getAij(Mij) {
    var N = Mij.length;
    var maxMij = 0;
    var maxIJ = [0, 1];

    for (var i = 0; i < N; i++) {
      for (var j = i + 1; j < N; j++) {
        if (Math.abs(maxMij) < Math.abs(Mij[i][j])) {
          maxMij = Math.abs(Mij[i][j]);
          maxIJ = [i, j];
        }
      }
    }

    return [maxIJ, maxMij];
  } // get max off-diagonal value from Upper Diagonal


  function getAijBig(Mij) {
    var N = Mij.length;
    var maxMij = 0;
    var maxIJ = [0, 1];

    for (var i = 0; i < N; i++) {
      for (var j = i + 1; j < N; j++) {
        if (abs(maxMij) < abs(Mij[i][j])) {
          maxMij = abs(Mij[i][j]);
          maxIJ = [i, j];
        }
      }
    }

    return [maxIJ, maxMij];
  } // sort results


  function sorting(E, S) {
    var N = E.length;
    var Ef = Array(N);
    var Sf = Array(N);

    for (var k = 0; k < N; k++) {
      Sf[k] = Array(N);
    }

    for (var i = 0; i < N; i++) {
      var minID = 0;
      var minE = E[0];

      for (var j = 0; j < E.length; j++) {
        if (E[j] < minE) {
          minID = j;
          minE = E[minID];
        }
      }

      Ef[i] = E.splice(minID, 1)[0];

      for (var _k5 = 0; _k5 < N; _k5++) {
        Sf[_k5][i] = S[_k5][minID];

        S[_k5].splice(minID, 1);
      }
    }

    return [clone(Ef), clone(Sf)];
  }
  /**
   * Create an array of a certain size and fill all items with an initial value
   * @param {number} size
   * @param {number} value
   * @return {number[]}
   */


  function createArray(size, value) {
    // TODO: as soon as all browsers support Array.fill, use that instead (IE doesn't support it)
    var array = new Array(size);

    for (var i = 0; i < size; i++) {
      array[i] = value;
    }

    return array;
  }

  return eigs;
});