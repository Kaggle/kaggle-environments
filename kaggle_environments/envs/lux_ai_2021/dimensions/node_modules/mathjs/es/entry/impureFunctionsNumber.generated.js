function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

/**
 * THIS FILE IS AUTO-GENERATED
 * DON'T MAKE CHANGES HERE
 */
import { config } from './configReadonly';
import { createChainClass, createChain, createNode, createArrayNode, createConditionalNode, createFunctionAssignmentNode, createObjectNode, createParenthesisNode, createRelationalNode, createReviver, createBlockNode, createOperatorNode, createSymbolNode, createAccessorNode, createConstantNode, createRangeNode, createAssignmentNode, createFunctionNode, createIndexNode, createParse, createEvaluate, createHelpClass, createSimplify, createRationalize, createCompile, createHelp, createParserClass, createDerivative, createParser, createFilterTransform, createMapTransform, createMinTransform, createForEachTransform, createSubsetTransform, createApplyTransform, createRangeTransform, createSumTransform, createMaxTransform, createMeanTransform, createStdTransform, createVarianceTransform } from '../factoriesNumber';
import { typed, Range, nthRoot, e, _false, LN10, LOG10E, _NaN, phi, SQRT1_2 // eslint-disable-line camelcase
, tau, version, string, filter, map, combinationsWithRep, pickRandom, randomInt, compare, compareText, smaller, larger, erf, max, format, clone, typeOf, unaryMinus, abs, cbrt, cube, expm1, floor, lcm, log2, multiplyScalar, sign, square, xgcd, pow, log1p, norm, bitAnd, bitOr, leftShift, rightLogShift, not, xor, matrix, combinations, acos, acot, acsc, asec, asin, atan, atanh, cosh, coth, csch, sech, sinh, tanh, isNegative, isZero, ResultSet, round, LN2, _null, SQRT2, number, forEach, size, random, compareNatural, equalText, largerEq, min, print, isNumeric, isPrime, replacer, addScalar, exp, gcd, mod, sqrt, divideScalar, add, bitNot, rightArithShift, or, subset, acosh, acsch, asinh, cos, csc, sin, isInteger, isNaN, catalan, _Infinity, pi, _true, apply, partitionSelect, equalScalar, smallerEq, unequal, sum, hasNumericValue, unaryPlus, fix, multiply, log, bitXor, index, acoth, atan2, sec, isPositive, hypot, composition, range, equal, mode, quantileSeq, numeric, log10, divide, gamma, cot, LOG2E, factorial, permutations, prod, median, ceil, and, tan, boolean as _boolean, multinomial, mean, subtract, deepEqual, variance, asech, stirlingS2, std, bellNumbers, mad } from './pureFunctionsNumber.generated';
var math = {}; // NOT pure!

var mathWithTransform = {}; // NOT pure!

var classes = {}; // NOT pure!

export var Chain = createChainClass({
  math: math
});
export var chain = createChain({
  Chain: Chain,
  typed: typed
});
export var Node = createNode({
  mathWithTransform: mathWithTransform
});
export var ArrayNode = createArrayNode({
  Node: Node
});
export var ConditionalNode = createConditionalNode({
  Node: Node
});
export var FunctionAssignmentNode = createFunctionAssignmentNode({
  Node: Node,
  typed: typed
});
export var ObjectNode = createObjectNode({
  Node: Node
});
export var ParenthesisNode = createParenthesisNode({
  Node: Node
});
export var RelationalNode = createRelationalNode({
  Node: Node
});
export var reviver = createReviver({
  classes: classes
});
export var BlockNode = createBlockNode({
  Node: Node,
  ResultSet: ResultSet
});
export var OperatorNode = createOperatorNode({
  Node: Node
});
export var SymbolNode = createSymbolNode({
  Node: Node,
  math: math
});
export var AccessorNode = createAccessorNode({
  Node: Node,
  subset: subset
});
export var ConstantNode = createConstantNode({
  Node: Node
});
export var RangeNode = createRangeNode({
  Node: Node
});
export var AssignmentNode = createAssignmentNode({
  matrix: matrix,
  Node: Node,
  subset: subset
});
export var FunctionNode = createFunctionNode({
  Node: Node,
  SymbolNode: SymbolNode,
  math: math
});
export var IndexNode = createIndexNode({
  Node: Node,
  Range: Range,
  size: size
});
export var parse = createParse({
  AccessorNode: AccessorNode,
  ArrayNode: ArrayNode,
  AssignmentNode: AssignmentNode,
  BlockNode: BlockNode,
  ConditionalNode: ConditionalNode,
  ConstantNode: ConstantNode,
  FunctionAssignmentNode: FunctionAssignmentNode,
  FunctionNode: FunctionNode,
  IndexNode: IndexNode,
  ObjectNode: ObjectNode,
  OperatorNode: OperatorNode,
  ParenthesisNode: ParenthesisNode,
  RangeNode: RangeNode,
  RelationalNode: RelationalNode,
  SymbolNode: SymbolNode,
  config: config,
  numeric: numeric,
  typed: typed
});
export var evaluate = createEvaluate({
  parse: parse,
  typed: typed
});
export var Help = createHelpClass({
  parse: parse
});
export var simplify = createSimplify({
  ConstantNode: ConstantNode,
  FunctionNode: FunctionNode,
  OperatorNode: OperatorNode,
  ParenthesisNode: ParenthesisNode,
  SymbolNode: SymbolNode,
  add: add,
  config: config,
  divide: divide,
  equal: equal,
  isZero: isZero,
  mathWithTransform: mathWithTransform,
  multiply: multiply,
  parse: parse,
  pow: pow,
  subtract: subtract,
  typed: typed
});
export var rationalize = createRationalize({
  ConstantNode: ConstantNode,
  FunctionNode: FunctionNode,
  OperatorNode: OperatorNode,
  ParenthesisNode: ParenthesisNode,
  SymbolNode: SymbolNode,
  add: add,
  config: config,
  divide: divide,
  equal: equal,
  isZero: isZero,
  mathWithTransform: mathWithTransform,
  multiply: multiply,
  parse: parse,
  pow: pow,
  simplify: simplify,
  subtract: subtract,
  typed: typed
});
export var compile = createCompile({
  parse: parse,
  typed: typed
});
export var help = createHelp({
  Help: Help,
  mathWithTransform: mathWithTransform,
  typed: typed
});
export var Parser = createParserClass({
  parse: parse
});
export var derivative = createDerivative({
  ConstantNode: ConstantNode,
  FunctionNode: FunctionNode,
  OperatorNode: OperatorNode,
  ParenthesisNode: ParenthesisNode,
  SymbolNode: SymbolNode,
  config: config,
  equal: equal,
  isZero: isZero,
  numeric: numeric,
  parse: parse,
  simplify: simplify,
  typed: typed
});
export var parser = createParser({
  Parser: Parser,
  typed: typed
});

_extends(math, {
  typed: typed,
  chain: chain,
  nthRoot: nthRoot,
  e: e,
  "false": _false,
  LN10: LN10,
  LOG10E: LOG10E,
  NaN: _NaN,
  phi: phi,
  SQRT1_2: SQRT1_2,
  tau: tau,
  version: version,
  string: string,
  filter: filter,
  map: map,
  combinationsWithRep: combinationsWithRep,
  pickRandom: pickRandom,
  randomInt: randomInt,
  compare: compare,
  compareText: compareText,
  smaller: smaller,
  larger: larger,
  erf: erf,
  max: max,
  format: format,
  clone: clone,
  typeOf: typeOf,
  reviver: reviver,
  unaryMinus: unaryMinus,
  abs: abs,
  cbrt: cbrt,
  cube: cube,
  expm1: expm1,
  floor: floor,
  lcm: lcm,
  log2: log2,
  multiplyScalar: multiplyScalar,
  sign: sign,
  square: square,
  xgcd: xgcd,
  pow: pow,
  log1p: log1p,
  norm: norm,
  bitAnd: bitAnd,
  bitOr: bitOr,
  leftShift: leftShift,
  rightLogShift: rightLogShift,
  not: not,
  xor: xor,
  matrix: matrix,
  combinations: combinations,
  acos: acos,
  acot: acot,
  acsc: acsc,
  asec: asec,
  asin: asin,
  atan: atan,
  atanh: atanh,
  cosh: cosh,
  coth: coth,
  csch: csch,
  sech: sech,
  sinh: sinh,
  tanh: tanh,
  isNegative: isNegative,
  isZero: isZero,
  round: round,
  'E': e,
  LN2: LN2,
  "null": _null,
  SQRT2: SQRT2,
  number: number,
  forEach: forEach,
  size: size,
  random: random,
  compareNatural: compareNatural,
  equalText: equalText,
  largerEq: largerEq,
  min: min,
  print: print,
  isNumeric: isNumeric,
  isPrime: isPrime,
  replacer: replacer,
  addScalar: addScalar,
  exp: exp,
  gcd: gcd,
  mod: mod,
  sqrt: sqrt,
  divideScalar: divideScalar,
  add: add,
  bitNot: bitNot,
  rightArithShift: rightArithShift,
  or: or,
  subset: subset,
  acosh: acosh,
  acsch: acsch,
  asinh: asinh,
  cos: cos,
  csc: csc,
  sin: sin,
  isInteger: isInteger,
  isNaN: isNaN,
  catalan: catalan,
  Infinity: _Infinity,
  pi: pi,
  "true": _true,
  apply: apply,
  partitionSelect: partitionSelect,
  equalScalar: equalScalar,
  smallerEq: smallerEq,
  unequal: unequal,
  sum: sum,
  hasNumericValue: hasNumericValue,
  unaryPlus: unaryPlus,
  fix: fix,
  multiply: multiply,
  log: log,
  bitXor: bitXor,
  index: index,
  acoth: acoth,
  atan2: atan2,
  sec: sec,
  isPositive: isPositive,
  hypot: hypot,
  composition: composition,
  'PI': pi,
  range: range,
  equal: equal,
  mode: mode,
  quantileSeq: quantileSeq,
  numeric: numeric,
  log10: log10,
  divide: divide,
  gamma: gamma,
  cot: cot,
  LOG2E: LOG2E,
  factorial: factorial,
  permutations: permutations,
  prod: prod,
  median: median,
  ceil: ceil,
  and: and,
  tan: tan,
  "boolean": _boolean,
  parse: parse,
  evaluate: evaluate,
  multinomial: multinomial,
  mean: mean,
  subtract: subtract,
  simplify: simplify,
  rationalize: rationalize,
  compile: compile,
  deepEqual: deepEqual,
  variance: variance,
  asech: asech,
  help: help,
  stirlingS2: stirlingS2,
  std: std,
  derivative: derivative,
  parser: parser,
  bellNumbers: bellNumbers,
  mad: mad,
  config: config
});

_extends(mathWithTransform, math, {
  filter: createFilterTransform({
    typed: typed
  }),
  map: createMapTransform({
    typed: typed
  }),
  min: createMinTransform({
    smaller: smaller,
    typed: typed
  }),
  forEach: createForEachTransform({
    typed: typed
  }),
  subset: createSubsetTransform({
    matrix: matrix,
    typed: typed
  }),
  apply: createApplyTransform({
    isInteger: isInteger,
    typed: typed
  }),
  range: createRangeTransform({
    matrix: matrix,
    config: config,
    larger: larger,
    largerEq: largerEq,
    smaller: smaller,
    smallerEq: smallerEq,
    typed: typed
  }),
  sum: createSumTransform({
    add: add,
    config: config,
    typed: typed
  }),
  max: createMaxTransform({
    larger: larger,
    typed: typed
  }),
  mean: createMeanTransform({
    add: add,
    divide: divide,
    typed: typed
  }),
  std: createStdTransform({
    sqrt: sqrt,
    typed: typed,
    variance: variance
  }),
  variance: createVarianceTransform({
    add: add,
    apply: apply,
    divide: divide,
    isNaN: isNaN,
    multiply: multiply,
    subtract: subtract,
    typed: typed
  })
});

_extends(classes, {
  Range: Range,
  Chain: Chain,
  Node: Node,
  ArrayNode: ArrayNode,
  ConditionalNode: ConditionalNode,
  FunctionAssignmentNode: FunctionAssignmentNode,
  ObjectNode: ObjectNode,
  ParenthesisNode: ParenthesisNode,
  RelationalNode: RelationalNode,
  ResultSet: ResultSet,
  BlockNode: BlockNode,
  OperatorNode: OperatorNode,
  SymbolNode: SymbolNode,
  AccessorNode: AccessorNode,
  ConstantNode: ConstantNode,
  RangeNode: RangeNode,
  AssignmentNode: AssignmentNode,
  FunctionNode: FunctionNode,
  IndexNode: IndexNode,
  Help: Help,
  Parser: Parser
});

Chain.createProxy(math);
export { embeddedDocs as docs } from '../expression/embeddedDocs/embeddedDocs';