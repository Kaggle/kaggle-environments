// TODO: deprecated since version 6.0.0. Date: 2019-04-14
// "deprecatedEval" is also exposed as "eval" in the code compiled to ES5+CommonJs
import { createDeprecatedEval } from '../expression/function/eval';
import { createDeprecatedImport } from '../core/function/deprecatedImport';
import { createDeprecatedVar } from '../function/statistics/variance';
import { createDeprecatedTypeof } from '../function/utils/typeOf';
import { isAccessorNode, isArray, isArrayNode, isAssignmentNode, isBigNumber, isBlockNode, isBoolean, isChain, isCollection, isComplex, isConditionalNode, isConstantNode, isDate, isDenseMatrix, isFraction, isFunction, isFunctionAssignmentNode, isFunctionNode, isHelp, isIndex, isIndexNode, isMatrix, isNode, isNull, isNumber, isObject, isObjectNode, isOperatorNode, isParenthesisNode, isRange, isRangeNode, isRegExp, isResultSet, isSparseMatrix, isString, isSymbolNode, isUndefined, isUnit } from '../utils/is';
import { ArgumentsError } from '../error/ArgumentsError';
import { DimensionError } from '../error/DimensionError';
import { IndexError } from '../error/IndexError';
import { lazy } from '../utils/object';
import { warnOnce } from '../utils/log';
import { BigNumber, Complex, DenseMatrix, FibonacciHeap, Fraction, ImmutableDenseMatrix, Index, Matrix, ResultSet, Range, Spa, SparseMatrix, typeOf, Unit, variance } from './pureFunctionsAny.generated';
import { AccessorNode, ArrayNode, AssignmentNode, BlockNode, Chain, ConditionalNode, ConstantNode, evaluate, FunctionAssignmentNode, FunctionNode, Help, IndexNode, Node, ObjectNode, OperatorNode, ParenthesisNode, parse, Parser, RangeNode, RelationalNode, reviver, SymbolNode } from './impureFunctionsAny.generated';
export var deprecatedEval = /* #__PURE__ */createDeprecatedEval({
  evaluate: evaluate
}); // "deprecatedImport" is also exposed as "import" in the code compiled to ES5+CommonJs

export var deprecatedImport = /* #__PURE__ */createDeprecatedImport({}); // "deprecatedVar" is also exposed as "var" in the code compiled to ES5+CommonJs

export var deprecatedVar = /* #__PURE__ */createDeprecatedVar({
  variance: variance
}); // "deprecatedTypeof" is also exposed as "typeof" in the code compiled to ES5+CommonJs

export var deprecatedTypeof = /* #__PURE__ */createDeprecatedTypeof({
  typeOf: typeOf
});
export var type = /* #__PURE__ */createDeprecatedProperties('type', {
  isNumber: isNumber,
  isComplex: isComplex,
  isBigNumber: isBigNumber,
  isFraction: isFraction,
  isUnit: isUnit,
  isString: isString,
  isArray: isArray,
  isMatrix: isMatrix,
  isCollection: isCollection,
  isDenseMatrix: isDenseMatrix,
  isSparseMatrix: isSparseMatrix,
  isRange: isRange,
  isIndex: isIndex,
  isBoolean: isBoolean,
  isResultSet: isResultSet,
  isHelp: isHelp,
  isFunction: isFunction,
  isDate: isDate,
  isRegExp: isRegExp,
  isObject: isObject,
  isNull: isNull,
  isUndefined: isUndefined,
  isAccessorNode: isAccessorNode,
  isArrayNode: isArrayNode,
  isAssignmentNode: isAssignmentNode,
  isBlockNode: isBlockNode,
  isConditionalNode: isConditionalNode,
  isConstantNode: isConstantNode,
  isFunctionAssignmentNode: isFunctionAssignmentNode,
  isFunctionNode: isFunctionNode,
  isIndexNode: isIndexNode,
  isNode: isNode,
  isObjectNode: isObjectNode,
  isOperatorNode: isOperatorNode,
  isParenthesisNode: isParenthesisNode,
  isRangeNode: isRangeNode,
  isSymbolNode: isSymbolNode,
  isChain: isChain,
  BigNumber: BigNumber,
  Chain: Chain,
  Complex: Complex,
  Fraction: Fraction,
  Matrix: Matrix,
  DenseMatrix: DenseMatrix,
  SparseMatrix: SparseMatrix,
  Spa: Spa,
  FibonacciHeap: FibonacciHeap,
  ImmutableDenseMatrix: ImmutableDenseMatrix,
  Index: Index,
  Range: Range,
  ResultSet: ResultSet,
  Unit: Unit,
  Help: Help,
  Parser: Parser
});
export var expression = /* #__PURE__ */createDeprecatedProperties('expression', {
  parse: parse,
  Parser: Parser,
  node: createDeprecatedProperties('expression.node', {
    AccessorNode: AccessorNode,
    ArrayNode: ArrayNode,
    AssignmentNode: AssignmentNode,
    BlockNode: BlockNode,
    ConditionalNode: ConditionalNode,
    ConstantNode: ConstantNode,
    IndexNode: IndexNode,
    FunctionAssignmentNode: FunctionAssignmentNode,
    FunctionNode: FunctionNode,
    Node: Node,
    ObjectNode: ObjectNode,
    OperatorNode: OperatorNode,
    ParenthesisNode: ParenthesisNode,
    RangeNode: RangeNode,
    RelationalNode: RelationalNode,
    SymbolNode: SymbolNode
  })
});
export var json = /* #__PURE__ */createDeprecatedProperties('json', {
  reviver: reviver
});
export var error = /* #__PURE__ */createDeprecatedProperties('error', {
  ArgumentsError: ArgumentsError,
  DimensionError: DimensionError,
  IndexError: IndexError
});

function createDeprecatedProperties(path, props) {
  var obj = {};
  Object.keys(props).forEach(function (name) {
    lazy(obj, name, function () {
      warnOnce("math.".concat(path, ".").concat(name, " is moved to math.").concat(name, " in v6.0.0. ") + 'Please use the new location instead.');
      return props[name];
    });
  });
  return obj;
}