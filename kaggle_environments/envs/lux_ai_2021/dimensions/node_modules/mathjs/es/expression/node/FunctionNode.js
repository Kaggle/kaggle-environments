function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

import { isAccessorNode, isFunctionAssignmentNode, isIndexNode, isNode, isSymbolNode } from '../../utils/is';
import { escape } from '../../utils/string';
import { hasOwnProperty } from '../../utils/object';
import { map } from '../../utils/array';
import { getSafeProperty, validateSafeMethod } from '../../utils/customs';
import { factory } from '../../utils/factory';
import { defaultTemplate, latexFunctions } from '../../utils/latex';
var name = 'FunctionNode';
var dependencies = ['math', 'Node', 'SymbolNode'];
export var createFunctionNode = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var math = _ref.math,
      Node = _ref.Node,
      SymbolNode = _ref.SymbolNode;

  /**
   * @constructor FunctionNode
   * @extends {./Node}
   * invoke a list with arguments on a node
   * @param {./Node | string} fn Node resolving with a function on which to invoke
   *                             the arguments, typically a SymboNode or AccessorNode
   * @param {./Node[]} args
   */
  function FunctionNode(fn, args) {
    if (!(this instanceof FunctionNode)) {
      throw new SyntaxError('Constructor must be called with the new operator');
    }

    if (typeof fn === 'string') {
      fn = new SymbolNode(fn);
    } // validate input


    if (!isNode(fn)) throw new TypeError('Node expected as parameter "fn"');

    if (!Array.isArray(args) || !args.every(isNode)) {
      throw new TypeError('Array containing Nodes expected for parameter "args"');
    }

    this.fn = fn;
    this.args = args || []; // readonly property name

    Object.defineProperty(this, 'name', {
      get: function () {
        return this.fn.name || '';
      }.bind(this),
      set: function set() {
        throw new Error('Cannot assign a new name, name is read-only');
      }
    }); // TODO: deprecated since v3, remove some day

    var deprecated = function deprecated() {
      throw new Error('Property `FunctionNode.object` is deprecated, use `FunctionNode.fn` instead');
    };

    Object.defineProperty(this, 'object', {
      get: deprecated,
      set: deprecated
    });
  }

  FunctionNode.prototype = new Node();
  FunctionNode.prototype.type = 'FunctionNode';
  FunctionNode.prototype.isFunctionNode = true;
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

  FunctionNode.prototype._compile = function (math, argNames) {
    if (!(this instanceof FunctionNode)) {
      throw new TypeError('No valid FunctionNode');
    } // compile arguments


    var evalArgs = map(this.args, function (arg) {
      return arg._compile(math, argNames);
    });

    if (isSymbolNode(this.fn)) {
      // we can statically determine whether the function has an rawArgs property
      var _name = this.fn.name;
      var fn = _name in math ? getSafeProperty(math, _name) : undefined;
      var isRaw = typeof fn === 'function' && fn.rawArgs === true;

      if (isRaw) {
        // pass unevaluated parameters (nodes) to the function
        // "raw" evaluation
        var rawArgs = this.args;
        return function evalFunctionNode(scope, args, context) {
          return (_name in scope ? getSafeProperty(scope, _name) : fn)(rawArgs, math, _extends({}, scope, args));
        };
      } else {
        // "regular" evaluation
        if (evalArgs.length === 1) {
          var evalArg0 = evalArgs[0];
          return function evalFunctionNode(scope, args, context) {
            return (_name in scope ? getSafeProperty(scope, _name) : fn)(evalArg0(scope, args, context));
          };
        } else if (evalArgs.length === 2) {
          var _evalArg = evalArgs[0];
          var evalArg1 = evalArgs[1];
          return function evalFunctionNode(scope, args, context) {
            return (_name in scope ? getSafeProperty(scope, _name) : fn)(_evalArg(scope, args, context), evalArg1(scope, args, context));
          };
        } else {
          return function evalFunctionNode(scope, args, context) {
            return (_name in scope ? getSafeProperty(scope, _name) : fn).apply(null, map(evalArgs, function (evalArg) {
              return evalArg(scope, args, context);
            }));
          };
        }
      }
    } else if (isAccessorNode(this.fn) && isIndexNode(this.fn.index) && this.fn.index.isObjectProperty()) {
      // execute the function with the right context: the object of the AccessorNode
      var evalObject = this.fn.object._compile(math, argNames);

      var prop = this.fn.index.getObjectProperty();
      var _rawArgs = this.args;
      return function evalFunctionNode(scope, args, context) {
        var object = evalObject(scope, args, context);
        validateSafeMethod(object, prop);
        var isRaw = object[prop] && object[prop].rawArgs;
        return isRaw ? object[prop](_rawArgs, math, _extends({}, scope, args)) // "raw" evaluation
        : object[prop].apply(object, map(evalArgs, function (evalArg) {
          // "regular" evaluation
          return evalArg(scope, args, context);
        }));
      };
    } else {
      // node.fn.isAccessorNode && !node.fn.index.isObjectProperty()
      // we have to dynamically determine whether the function has a rawArgs property
      var evalFn = this.fn._compile(math, argNames);

      var _rawArgs2 = this.args;
      return function evalFunctionNode(scope, args, context) {
        var fn = evalFn(scope, args, context);
        var isRaw = fn && fn.rawArgs;
        return isRaw ? fn(_rawArgs2, math, _extends({}, scope, args)) // "raw" evaluation
        : fn.apply(fn, map(evalArgs, function (evalArg) {
          // "regular" evaluation
          return evalArg(scope, args, context);
        }));
      };
    }
  };
  /**
   * Execute a callback for each of the child nodes of this node
   * @param {function(child: Node, path: string, parent: Node)} callback
   */


  FunctionNode.prototype.forEach = function (callback) {
    callback(this.fn, 'fn', this);

    for (var i = 0; i < this.args.length; i++) {
      callback(this.args[i], 'args[' + i + ']', this);
    }
  };
  /**
   * Create a new FunctionNode having it's childs be the results of calling
   * the provided callback function for each of the childs of the original node.
   * @param {function(child: Node, path: string, parent: Node): Node} callback
   * @returns {FunctionNode} Returns a transformed copy of the node
   */


  FunctionNode.prototype.map = function (callback) {
    var fn = this._ifNode(callback(this.fn, 'fn', this));

    var args = [];

    for (var i = 0; i < this.args.length; i++) {
      args[i] = this._ifNode(callback(this.args[i], 'args[' + i + ']', this));
    }

    return new FunctionNode(fn, args);
  };
  /**
   * Create a clone of this node, a shallow copy
   * @return {FunctionNode}
   */


  FunctionNode.prototype.clone = function () {
    return new FunctionNode(this.fn, this.args.slice(0));
  }; // backup Node's toString function
  // @private


  var nodeToString = FunctionNode.prototype.toString;
  /**
   * Get string representation. (wrapper function)
   * This overrides parts of Node's toString function.
   * If callback is an object containing callbacks, it
   * calls the correct callback for the current node,
   * otherwise it falls back to calling Node's toString
   * function.
   *
   * @param {Object} options
   * @return {string} str
   * @override
   */

  FunctionNode.prototype.toString = function (options) {
    var customString;
    var name = this.fn.toString(options);

    if (options && _typeof(options.handler) === 'object' && hasOwnProperty(options.handler, name)) {
      // callback is a map of callback functions
      customString = options.handler[name](this, options);
    }

    if (typeof customString !== 'undefined') {
      return customString;
    } // fall back to Node's toString


    return nodeToString.call(this, options);
  };
  /**
   * Get string representation
   * @param {Object} options
   * @return {string} str
   */


  FunctionNode.prototype._toString = function (options) {
    var args = this.args.map(function (arg) {
      return arg.toString(options);
    });
    var fn = isFunctionAssignmentNode(this.fn) ? '(' + this.fn.toString(options) + ')' : this.fn.toString(options); // format the arguments like "add(2, 4.2)"

    return fn + '(' + args.join(', ') + ')';
  };
  /**
   * Get a JSON representation of the node
   * @returns {Object}
   */


  FunctionNode.prototype.toJSON = function () {
    return {
      mathjs: 'FunctionNode',
      fn: this.fn,
      args: this.args
    };
  };
  /**
   * Instantiate an AssignmentNode from its JSON representation
   * @param {Object} json  An object structured like
   *                       `{"mathjs": "FunctionNode", fn: ..., args: ...}`,
   *                       where mathjs is optional
   * @returns {FunctionNode}
   */


  FunctionNode.fromJSON = function (json) {
    return new FunctionNode(json.fn, json.args);
  };
  /**
   * Get HTML representation
   * @param {Object} options
   * @return {string} str
   */


  FunctionNode.prototype.toHTML = function (options) {
    var args = this.args.map(function (arg) {
      return arg.toHTML(options);
    }); // format the arguments like "add(2, 4.2)"

    return '<span class="math-function">' + escape(this.fn) + '</span><span class="math-paranthesis math-round-parenthesis">(</span>' + args.join('<span class="math-separator">,</span>') + '<span class="math-paranthesis math-round-parenthesis">)</span>';
  };
  /*
   * Expand a LaTeX template
   *
   * @param {string} template
   * @param {Node} node
   * @param {Object} options
   * @private
   **/


  function expandTemplate(template, node, options) {
    var latex = ''; // Match everything of the form ${identifier} or ${identifier[2]} or $$
    // while submatching identifier and 2 (in the second case)

    var regex = new RegExp('\\$(?:\\{([a-z_][a-z_0-9]*)(?:\\[([0-9]+)\\])?\\}|\\$)', 'ig');
    var inputPos = 0; // position in the input string

    var match;

    while ((match = regex.exec(template)) !== null) {
      // go through all matches
      // add everything in front of the match to the LaTeX string
      latex += template.substring(inputPos, match.index);
      inputPos = match.index;

      if (match[0] === '$$') {
        // escaped dollar sign
        latex += '$';
        inputPos++;
      } else {
        // template parameter
        inputPos += match[0].length;
        var property = node[match[1]];

        if (!property) {
          throw new ReferenceError('Template: Property ' + match[1] + ' does not exist.');
        }

        if (match[2] === undefined) {
          // no square brackets
          switch (_typeof(property)) {
            case 'string':
              latex += property;
              break;

            case 'object':
              if (isNode(property)) {
                latex += property.toTex(options);
              } else if (Array.isArray(property)) {
                // make array of Nodes into comma separated list
                latex += property.map(function (arg, index) {
                  if (isNode(arg)) {
                    return arg.toTex(options);
                  }

                  throw new TypeError('Template: ' + match[1] + '[' + index + '] is not a Node.');
                }).join(',');
              } else {
                throw new TypeError('Template: ' + match[1] + ' has to be a Node, String or array of Nodes');
              }

              break;

            default:
              throw new TypeError('Template: ' + match[1] + ' has to be a Node, String or array of Nodes');
          }
        } else {
          // with square brackets
          if (isNode(property[match[2]] && property[match[2]])) {
            latex += property[match[2]].toTex(options);
          } else {
            throw new TypeError('Template: ' + match[1] + '[' + match[2] + '] is not a Node.');
          }
        }
      }
    }

    latex += template.slice(inputPos); // append rest of the template

    return latex;
  } // backup Node's toTex function
  // @private


  var nodeToTex = FunctionNode.prototype.toTex;
  /**
   * Get LaTeX representation. (wrapper function)
   * This overrides parts of Node's toTex function.
   * If callback is an object containing callbacks, it
   * calls the correct callback for the current node,
   * otherwise it falls back to calling Node's toTex
   * function.
   *
   * @param {Object} options
   * @return {string}
   */

  FunctionNode.prototype.toTex = function (options) {
    var customTex;

    if (options && _typeof(options.handler) === 'object' && hasOwnProperty(options.handler, this.name)) {
      // callback is a map of callback functions
      customTex = options.handler[this.name](this, options);
    }

    if (typeof customTex !== 'undefined') {
      return customTex;
    } // fall back to Node's toTex


    return nodeToTex.call(this, options);
  };
  /**
   * Get LaTeX representation
   * @param {Object} options
   * @return {string} str
   */


  FunctionNode.prototype._toTex = function (options) {
    var args = this.args.map(function (arg) {
      // get LaTeX of the arguments
      return arg.toTex(options);
    });
    var latexConverter;

    if (latexFunctions[this.name]) {
      latexConverter = latexFunctions[this.name];
    } // toTex property on the function itself


    if (math[this.name] && (typeof math[this.name].toTex === 'function' || _typeof(math[this.name].toTex) === 'object' || typeof math[this.name].toTex === 'string')) {
      // .toTex is a callback function
      latexConverter = math[this.name].toTex;
    }

    var customToTex;

    switch (_typeof(latexConverter)) {
      case 'function':
        // a callback function
        customToTex = latexConverter(this, options);
        break;

      case 'string':
        // a template string
        customToTex = expandTemplate(latexConverter, this, options);
        break;

      case 'object':
        // an object with different "converters" for different numbers of arguments
        switch (_typeof(latexConverter[args.length])) {
          case 'function':
            customToTex = latexConverter[args.length](this, options);
            break;

          case 'string':
            customToTex = expandTemplate(latexConverter[args.length], this, options);
            break;
        }

    }

    if (typeof customToTex !== 'undefined') {
      return customToTex;
    }

    return expandTemplate(defaultTemplate, this, options);
  };
  /**
   * Get identifier.
   * @return {string}
   */


  FunctionNode.prototype.getIdentifier = function () {
    return this.type + ':' + this.name;
  };

  return FunctionNode;
}, {
  isClass: true,
  isNode: true
});