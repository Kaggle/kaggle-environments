/**
 * @preserve date-and-time.js locale configuration
 * @preserve Hungarian (hu)
 * @preserve It is using moment.js locale configuration as a reference.
 */
(function (global) {
    'use strict';

    var exec = function (date) {
        var code = 'hu';

        date.locale(code, {
            res: {
                MMMM: ['január', 'február', 'március', 'április', 'május', 'június', 'július', 'augusztus', 'szeptember', 'október', 'november', 'december'],
                MMM: ['jan', 'feb', 'márc', 'ápr', 'máj', 'jún', 'júl', 'aug', 'szept', 'okt', 'nov', 'dec'],
                dddd: ['vasárnap', 'hétfő', 'kedd', 'szerda', 'csütörtök', 'péntek', 'szombat'],
                ddd: ['vas', 'hét', 'kedd', 'sze', 'csüt', 'pén', 'szo'],
                dd: ['v', 'h', 'k', 'sze', 'cs', 'p', 'szo'],
                A: ['de', 'du']
            }
        });
        return code;
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
