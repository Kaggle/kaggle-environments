function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

import { isBigNumber, isConstantNode, isNode, isRangeNode, isSymbolNode } from '../../utils/is';
import { map } from '../../utils/array';
import { escape } from '../../utils/string';
import { factory } from '../../utils/factory';
import { getSafeProperty } from '../../utils/customs';
var name = 'IndexNode';
var dependencies = ['Range', 'Node', 'size'];
export var createIndexNode = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var Range = _ref.Range,
      Node = _ref.Node,
      size = _ref.size;

  /**
   * @constructor IndexNode
   * @extends Node
   *
   * Describes a subset of a matrix or an object property.
   * Cannot be used on its own, needs to be used within an AccessorNode or
   * AssignmentNode.
   *
   * @param {Node[]} dimensions
   * @param {boolean} [dotNotation=false]  Optional property describing whether
   *                                       this index was written using dot
   *                                       notation like `a.b`, or using bracket
   *                                       notation like `a["b"]` (default).
   *                                       Used to stringify an IndexNode.
   */
  function IndexNode(dimensions, dotNotation) {
    if (!(this instanceof IndexNode)) {
      throw new SyntaxError('Constructor must be called with the new operator');
    }

    this.dimensions = dimensions;
    this.dotNotation = dotNotation || false; // validate input

    if (!Array.isArray(dimensions) || !dimensions.every(isNode)) {
      throw new TypeError('Array containing Nodes expected for parameter "dimensions"');
    }

    if (this.dotNotation && !this.isObjectProperty()) {
      throw new Error('dotNotation only applicable for object properties');
    } // TODO: deprecated since v3, remove some day


    var deprecated = function deprecated() {
      throw new Error('Property `IndexNode.object` is deprecated, use `IndexNode.fn` instead');
    };

    Object.defineProperty(this, 'object', {
      get: deprecated,
      set: deprecated
    });
  }

  IndexNode.prototype = new Node();
  IndexNode.prototype.type = 'IndexNode';
  IndexNode.prototype.isIndexNode = true;
  /**
   * Compile a node into a JavaScript function.
   * This basically pre-calculates as much as possible and only leaves open
   * calculations which depend on a dynamic scope with variables.
   * @param {Object} math     Math.js namespace with functions and constants.
   * @param {Object} argNames An object with argument names as key and `true`
   *                          as value. Used in the SymbolNode to optimize
   *                          for arguments from user assigned functions
   *                          (see FunctionAssignmentNode) or special symbols
   *                          like `end` (see IndexNode).
   * @return {function} Returns a function which can be called like:
   *                        evalNode(scope: Object, args: Object, context: *)
   */

  IndexNode.prototype._compile = function (math, argNames) {
    // TODO: implement support for bignumber (currently bignumbers are silently
    //       reduced to numbers when changing the value to zero-based)
    // TODO: Optimization: when the range values are ConstantNodes,
    //       we can beforehand resolve the zero-based value
    // optimization for a simple object property
    var evalDimensions = map(this.dimensions, function (range, i) {
      if (isRangeNode(range)) {
        if (range.needsEnd()) {
          // create a range containing end (like '4:end')
          var childArgNames = Object.create(argNames);
          childArgNames.end = true;

          var evalStart = range.start._compile(math, childArgNames);

          var evalEnd = range.end._compile(math, childArgNames);

          var evalStep = range.step ? range.step._compile(math, childArgNames) : function () {
            return 1;
          };
          return function evalDimension(scope, args, context) {
            var s = size(context).valueOf();
            var childArgs = Object.create(args);
            childArgs.end = s[i];
            return createRange(evalStart(scope, childArgs, context), evalEnd(scope, childArgs, context), evalStep(scope, childArgs, context));
          };
        } else {
          // create range
          var _evalStart = range.start._compile(math, argNames);

          var _evalEnd = range.end._compile(math, argNames);

          var _evalStep = range.step ? range.step._compile(math, argNames) : function () {
            return 1;
          };

          return function evalDimension(scope, args, context) {
            return createRange(_evalStart(scope, args, context), _evalEnd(scope, args, context), _evalStep(scope, args, context));
          };
        }
      } else if (isSymbolNode(range) && range.name === 'end') {
        // SymbolNode 'end'
        var _childArgNames = Object.create(argNames);

        _childArgNames.end = true;

        var evalRange = range._compile(math, _childArgNames);

        return function evalDimension(scope, args, context) {
          var s = size(context).valueOf();
          var childArgs = Object.create(args);
          childArgs.end = s[i];
          return evalRange(scope, childArgs, context);
        };
      } else {
        // ConstantNode
        var _evalRange = range._compile(math, argNames);

        return function evalDimension(scope, args, context) {
          return _evalRange(scope, args, context);
        };
      }
    });
    var index = getSafeProperty(math, 'index');
    return function evalIndexNode(scope, args, context) {
      var dimensions = map(evalDimensions, function (evalDimension) {
        return evalDimension(scope, args, context);
      });
      return index.apply(void 0, _toConsumableArray(dimensions));
    };
  };
  /**
   * Execute a callback for each of the child nodes of this node
   * @param {function(child: Node, path: string, parent: Node)} callback
   */


  IndexNode.prototype.forEach = function (callback) {
    for (var i = 0; i < this.dimensions.length; i++) {
      callback(this.dimensions[i], 'dimensions[' + i + ']', this);
    }
  };
  /**
   * Create a new IndexNode having it's childs be the results of calling
   * the provided callback function for each of the childs of the original node.
   * @param {function(child: Node, path: string, parent: Node): Node} callback
   * @returns {IndexNode} Returns a transformed copy of the node
   */


  IndexNode.prototype.map = function (callback) {
    var dimensions = [];

    for (var i = 0; i < this.dimensions.length; i++) {
      dimensions[i] = this._ifNode(callback(this.dimensions[i], 'dimensions[' + i + ']', this));
    }

    return new IndexNode(dimensions, this.dotNotation);
  };
  /**
   * Create a clone of this node, a shallow copy
   * @return {IndexNode}
   */


  IndexNode.prototype.clone = function () {
    return new IndexNode(this.dimensions.slice(0), this.dotNotation);
  };
  /**
   * Test whether this IndexNode contains a single property name
   * @return {boolean}
   */


  IndexNode.prototype.isObjectProperty = function () {
    return this.dimensions.length === 1 && isConstantNode(this.dimensions[0]) && typeof this.dimensions[0].value === 'string';
  };
  /**
   * Returns the property name if IndexNode contains a property.
   * If not, returns null.
   * @return {string | null}
   */


  IndexNode.prototype.getObjectProperty = function () {
    return this.isObjectProperty() ? this.dimensions[0].value : null;
  };
  /**
   * Get string representation
   * @param {Object} options
   * @return {string} str
   */


  IndexNode.prototype._toString = function (options) {
    // format the parameters like "[1, 0:5]"
    return this.dotNotation ? '.' + this.getObjectProperty() : '[' + this.dimensions.join(', ') + ']';
  };
  /**
   * Get a JSON representation of the node
   * @returns {Object}
   */


  IndexNode.prototype.toJSON = function () {
    return {
      mathjs: 'IndexNode',
      dimensions: this.dimensions,
      dotNotation: this.dotNotation
    };
  };
  /**
   * Instantiate an IndexNode from its JSON representation
   * @param {Object} json  An object structured like
   *                       `{"mathjs": "IndexNode", dimensions: [...], dotNotation: false}`,
   *                       where mathjs is optional
   * @returns {IndexNode}
   */


  IndexNode.fromJSON = function (json) {
    return new IndexNode(json.dimensions, json.dotNotation);
  };
  /**
   * Get HTML representation
   * @param {Object} options
   * @return {string} str
   */


  IndexNode.prototype.toHTML = function (options) {
    // format the parameters like "[1, 0:5]"
    var dimensions = [];

    for (var i = 0; i < this.dimensions.length; i++) {
      dimensions[i] = this.dimensions[i].toHTML();
    }

    if (this.dotNotation) {
      return '<span class="math-operator math-accessor-operator">.</span>' + '<span class="math-symbol math-property">' + escape(this.getObjectProperty()) + '</span>';
    } else {
      return '<span class="math-parenthesis math-square-parenthesis">[</span>' + dimensions.join('<span class="math-separator">,</span>') + '<span class="math-parenthesis math-square-parenthesis">]</span>';
    }
  };
  /**
   * Get LaTeX representation
   * @param {Object} options
   * @return {string} str
   */


  IndexNode.prototype._toTex = function (options) {
    var dimensions = this.dimensions.map(function (range) {
      return range.toTex(options);
    });
    return this.dotNotation ? '.' + this.getObjectProperty() + '' : '_{' + dimensions.join(',') + '}';
  }; // helper function to create a Range from start, step and end


  function createRange(start, end, step) {
    return new Range(isBigNumber(start) ? start.toNumber() : start, isBigNumber(end) ? end.toNumber() : end, isBigNumber(step) ? step.toNumber() : step);
  }

  return IndexNode;
}, {
  isClass: true,
  isNode: true
});