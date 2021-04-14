// For backward compatibility, deprecated since version 6.0.0. Date: 2018-12-05
import { factory } from '../../utils/factory';
import { warnOnce } from '../../utils/log';
export var createDeprecatedEval = /* #__PURE__ */factory('eval', ['evaluate'], function (_ref) {
  var evaluate = _ref.evaluate;
  return function () {
    warnOnce('Function "eval" has been renamed to "evaluate" in v6.0.0, please use the new function instead.');

    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    return evaluate.apply(evaluate, args);
  };
});