/**
 * @preserve date-and-time.js locale configuration
 * @preserve Englis (en)
 * @preserve This is a dummy module.
 */
(function (global) {
    'use strict';

    var exec = function () {
        return 'en';
    };

    if (typeof module === 'object' && typeof module.exports === 'object') {
        (module.paths || []).push('./');
        module.exports = exec;
        exec(require('date-and-time'));
    } else if (typeof define === 'function' && define.amd) {
        define(['date-and-time'], exec);
    } else {
        exec(global.date);
    }

}(this));
