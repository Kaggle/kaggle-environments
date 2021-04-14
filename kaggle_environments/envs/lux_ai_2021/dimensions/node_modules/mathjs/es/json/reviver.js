import { factory } from '../utils/factory';
var name = 'reviver';
var dependencies = ['classes'];
export var createReviver = /* #__PURE__ */factory(name, dependencies, function (_ref) {
  var classes = _ref.classes;

  /**
   * Instantiate mathjs data types from their JSON representation
   * @param {string} key
   * @param {*} value
   * @returns {*} Returns the revived object
   */
  return function reviver(key, value) {
    var constructor = classes[value && value.mathjs];

    if (constructor && typeof constructor.fromJSON === 'function') {
      return constructor.fromJSON(value);
    }

    return value;
  };
});