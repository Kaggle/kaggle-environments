var all = require('./factoriesAny');

var _require = require('./core/create'),
    create = _require.create;

var defaultInstance = create(all); // TODO: not nice having to revert to CommonJS, find an ES6 solution

module.exports = /* #__PURE__ */defaultInstance;