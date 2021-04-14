import { factory } from './utils/factory';
import { version } from './version';
import { createBigNumberE, createBigNumberPhi, createBigNumberPi, createBigNumberTau } from './utils/bignumber/constants';
import { pi, tau, e, phi } from './plain/number';
export var createTrue = /* #__PURE__ */factory('true', [], function () {
  return true;
});
export var createFalse = /* #__PURE__ */factory('false', [], function () {
  return false;
});
export var createNull = /* #__PURE__ */factory('null', [], function () {
  return null;
});
export var createInfinity = /* #__PURE__ */recreateFactory('Infinity', ['config', '?BigNumber'], function (_ref) {
  var config = _ref.config,
      BigNumber = _ref.BigNumber;
  return config.number === 'BigNumber' ? new BigNumber(Infinity) : Infinity;
});
export var createNaN = /* #__PURE__ */recreateFactory('NaN', ['config', '?BigNumber'], function (_ref2) {
  var config = _ref2.config,
      BigNumber = _ref2.BigNumber;
  return config.number === 'BigNumber' ? new BigNumber(NaN) : NaN;
});
export var createPi = /* #__PURE__ */recreateFactory('pi', ['config', '?BigNumber'], function (_ref3) {
  var config = _ref3.config,
      BigNumber = _ref3.BigNumber;
  return config.number === 'BigNumber' ? createBigNumberPi(BigNumber) : pi;
});
export var createTau = /* #__PURE__ */recreateFactory('tau', ['config', '?BigNumber'], function (_ref4) {
  var config = _ref4.config,
      BigNumber = _ref4.BigNumber;
  return config.number === 'BigNumber' ? createBigNumberTau(BigNumber) : tau;
});
export var createE = /* #__PURE__ */recreateFactory('e', ['config', '?BigNumber'], function (_ref5) {
  var config = _ref5.config,
      BigNumber = _ref5.BigNumber;
  return config.number === 'BigNumber' ? createBigNumberE(BigNumber) : e;
}); // golden ratio, (1+sqrt(5))/2

export var createPhi = /* #__PURE__ */recreateFactory('phi', ['config', '?BigNumber'], function (_ref6) {
  var config = _ref6.config,
      BigNumber = _ref6.BigNumber;
  return config.number === 'BigNumber' ? createBigNumberPhi(BigNumber) : phi;
});
export var createLN2 = /* #__PURE__ */recreateFactory('LN2', ['config', '?BigNumber'], function (_ref7) {
  var config = _ref7.config,
      BigNumber = _ref7.BigNumber;
  return config.number === 'BigNumber' ? new BigNumber(2).ln() : Math.LN2;
});
export var createLN10 = /* #__PURE__ */recreateFactory('LN10', ['config', '?BigNumber'], function (_ref8) {
  var config = _ref8.config,
      BigNumber = _ref8.BigNumber;
  return config.number === 'BigNumber' ? new BigNumber(10).ln() : Math.LN10;
});
export var createLOG2E = /* #__PURE__ */recreateFactory('LOG2E', ['config', '?BigNumber'], function (_ref9) {
  var config = _ref9.config,
      BigNumber = _ref9.BigNumber;
  return config.number === 'BigNumber' ? new BigNumber(1).div(new BigNumber(2).ln()) : Math.LOG2E;
});
export var createLOG10E = /* #__PURE__ */recreateFactory('LOG10E', ['config', '?BigNumber'], function (_ref10) {
  var config = _ref10.config,
      BigNumber = _ref10.BigNumber;
  return config.number === 'BigNumber' ? new BigNumber(1).div(new BigNumber(10).ln()) : Math.LOG10E;
});
export var createSQRT1_2 = /* #__PURE__ */recreateFactory( // eslint-disable-line camelcase
'SQRT1_2', ['config', '?BigNumber'], function (_ref11) {
  var config = _ref11.config,
      BigNumber = _ref11.BigNumber;
  return config.number === 'BigNumber' ? new BigNumber('0.5').sqrt() : Math.SQRT1_2;
});
export var createSQRT2 = /* #__PURE__ */recreateFactory('SQRT2', ['config', '?BigNumber'], function (_ref12) {
  var config = _ref12.config,
      BigNumber = _ref12.BigNumber;
  return config.number === 'BigNumber' ? new BigNumber(2).sqrt() : Math.SQRT2;
});
export var createI = /* #__PURE__ */recreateFactory('i', ['Complex'], function (_ref13) {
  var Complex = _ref13.Complex;
  return Complex.I;
}); // for backward compatibility with v5

export var createUppercasePi = /* #__PURE__ */factory('PI', ['pi'], function (_ref14) {
  var pi = _ref14.pi;
  return pi;
});
export var createUppercaseE = /* #__PURE__ */factory('E', ['e'], function (_ref15) {
  var e = _ref15.e;
  return e;
});
export var createVersion = /* #__PURE__ */factory('version', [], function () {
  return version;
}); // helper function to create a factory with a flag recreateOnConfigChange
// idea: allow passing optional properties to be attached to the factory function as 4th argument?

function recreateFactory(name, dependencies, create) {
  return factory(name, dependencies, create, {
    recreateOnConfigChange: true
  });
}