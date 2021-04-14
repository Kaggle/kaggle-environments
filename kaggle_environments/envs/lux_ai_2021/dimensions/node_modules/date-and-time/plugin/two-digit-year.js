(function (global) {
    'use strict';

    var exec = function (date) {
        var name = 'two-digit-year';

        date.plugin(name, {
            parser: {
                YY: function (str) {
                    var result = this.exec(/^\d\d/, str);
                    result.value += result.value < 70 ? 2000 : 1900;
                    return result;
                },
                Y: function (str) {
                    var result = this.exec(/^\d\d?/, str);
                    result.value += result.value < 70 ? 2000 : 1900;
                    return result;
                }
            }
        });
        return name;
    };

    if (typeof module === 'object' && typeof module.exports === 'object') {
        (module.paths || []).push('./');
        module.exports = exec;
        // This line will be removed in the next version.
        exec(require('date-and-time'));
    } else if (typeof define === 'function' && define.amd) {
        define(['date-and-time'], exec);
    } else {
        exec(global.date);
    }

}(this));
