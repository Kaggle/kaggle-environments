(function (global) {
    'use strict';

    var exec = function (date) {
        var name = 'microsecond';

        date.plugin('microsecond', {
            parser: {
                SSSSSS: function (str) {
                    var result = this.exec(/^\d{1,6}/, str);
                    result.value = result.value / 1000 | 0;
                    return result;
                },
                SSSSS: function (str) {
                    var result = this.exec(/^\d{1,5}/, str);
                    result.value = result.value / 100 | 0;
                    return result;
                },
                SSSS: function (str) {
                    var result = this.exec(/^\d{1,4}/, str);
                    result.value = result.value / 10 | 0;
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
