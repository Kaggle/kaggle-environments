function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

// function utils

/**
 * Memoize a given function by caching the computed result.
 * The cache of a memoized function can be cleared by deleting the `cache`
 * property of the function.
 *
 * @param {function} fn                     The function to be memoized.
 *                                          Must be a pure function.
 * @param {function(args: Array)} [hasher]  A custom hash builder.
 *                                          Is JSON.stringify by default.
 * @return {function}                       Returns the memoized function
 */
export function memoize(fn, hasher) {
  return function memoize() {
    if (_typeof(memoize.cache) !== 'object') {
      memoize.cache = {};
    }

    var args = [];

    for (var i = 0; i < arguments.length; i++) {
      args[i] = arguments[i];
    }

    var hash = hasher ? hasher(args) : JSON.stringify(args);

    if (!(hash in memoize.cache)) {
      memoize.cache[hash] = fn.apply(fn, args);
    }

    return memoize.cache[hash];
  };
}
/**
 * Memoize a given function by caching all results and the arguments,
 * and comparing against the arguments of previous results before
 * executing again.
 * This is less performant than `memoize` which calculates a hash,
 * which is very fast to compare. Use `memoizeCompare` only when it is
 * not possible to create a unique serializable hash from the function
 * arguments.
 * The isEqual function must compare two sets of arguments
 * and return true when equal (can be a deep equality check for example).
 * @param {function} fn
 * @param {function(a: *, b: *) : boolean} isEqual
 * @returns {function}
 */

export function memoizeCompare(fn, isEqual) {
  var memoize = function memoize() {
    var args = [];

    for (var i = 0; i < arguments.length; i++) {
      args[i] = arguments[i];
    }

    for (var c = 0; c < memoize.cache.length; c++) {
      var cached = memoize.cache[c];

      if (isEqual(args, cached.args)) {
        // TODO: move this cache entry to the top so recently used entries move up?
        return cached.res;
      }
    }

    var res = fn.apply(fn, args);
    memoize.cache.unshift({
      args: args,
      res: res
    });
    return res;
  };

  memoize.cache = [];
  return memoize;
}
/**
 * Find the maximum number of arguments expected by a typed function.
 * @param {function} fn   A typed function
 * @return {number} Returns the maximum number of expected arguments.
 *                  Returns -1 when no signatures where found on the function.
 */

export function maxArgumentCount(fn) {
  return Object.keys(fn.signatures || {}).reduce(function (args, signature) {
    var count = (signature.match(/,/g) || []).length + 1;
    return Math.max(args, count);
  }, -1);
}