(function (global) {
    'use strict';

    var exec = function (date) {
        var name = 'day-of-week';

        date.plugin(name, {
            parser: {
                dddd: function (str) { return this.find(this.res.dddd, str); },
                ddd: function (str) { return this.find(this.res.ddd, str); },
                dd: function (str) { return this.find(this.res.dd, str); }
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
