(function (global) {
    'use strict';

    var exec = function (date) {
        var name = 'ordinal';

        date.plugin(name, {
            formatter: {
                DDD: function (d) {
                    var day = d.getDate();

                    switch (day) {
                    case 1:
                    case 21:
                    case 31:
                        return day + 'st';
                    case 2:
                    case 22:
                        return day + 'nd';
                    case 3:
                    case 23:
                        return day + 'rd';
                    default:
                        return day + 'th';
                    }
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
