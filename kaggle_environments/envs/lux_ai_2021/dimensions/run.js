"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __asyncValues = (this && this.__asyncValues) || function (o) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var m = o[Symbol.asyncIterator], i;
    return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i);
    function verb(n) { i[n] = o[n] && function (v) { return new Promise(function (resolve, reject) { v = o[n](v), settle(resolve, reject, v.done, v.value); }); }; }
    function settle(resolve, reject, d, v) { Promise.resolve(v).then(function(v) { resolve({ value: v, done: d }); }, reject); }
};
exports.__esModule = true;
var dimensions_ai_1 = require("dimensions-ai");
var readline_1 = require("readline");
var _2020_challenge_1 = require("@lux-ai/2020-challenge");
var haliteLuxDesign = new _2020_challenge_1.LuxDesign("RPS");
var myDimension = dimensions_ai_1.create(haliteLuxDesign, {
    name: "Halite Lux",
    loggingLevel: dimensions_ai_1.Logger.LEVEL.NONE,
    activateStation: false,
    observe: false
});
var rl = readline_1["default"].createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});
var main = function () { return __awaiter(void 0, void 0, void 0, function () {
    var match, rl_1, rl_1_1, line, json, e_1_1;
    var e_1, _a;
    return __generator(this, function (_b) {
        switch (_b.label) {
            case 0:
                match = null;
                _b.label = 1;
            case 1:
                _b.trys.push([1, 9, 10, 15]);
                rl_1 = __asyncValues(rl);
                _b.label = 2;
            case 2: return [4 /*yield*/, rl_1.next()];
            case 3:
                if (!(rl_1_1 = _b.sent(), !rl_1_1.done)) return [3 /*break*/, 8];
                line = rl_1_1.value;
                json = JSON.parse(line);
                if (!(json.type && json.type === "start")) return [3 /*break*/, 5];
                return [4 /*yield*/, myDimension.createMatch([
                        {
                            file: "blank",
                            name: "bot1"
                        },
                        {
                            file: "blank",
                            name: "bot2"
                        },
                    ], {
                        detached: true,
                        agentOptions: { detached: true },
                        storeReplay: false,
                        storeErrorLogs: false
                    })];
            case 4:
                match = _b.sent();
                return [3 /*break*/, 7];
            case 5:
                if (!json.length) return [3 /*break*/, 7];
                // perform a step in the match
                console.log(json);
                return [4 /*yield*/, match.step([
                        { agentID: 0, command: mapNumToRPS(json[0].action) },
                        { agentID: 1, command: mapNumToRPS(json[1].action) },
                    ])];
            case 6:
                _b.sent();
                _b.label = 7;
            case 7: return [3 /*break*/, 2];
            case 8: return [3 /*break*/, 15];
            case 9:
                e_1_1 = _b.sent();
                e_1 = { error: e_1_1 };
                return [3 /*break*/, 15];
            case 10:
                _b.trys.push([10, , 13, 14]);
                if (!(rl_1_1 && !rl_1_1.done && (_a = rl_1["return"]))) return [3 /*break*/, 12];
                return [4 /*yield*/, _a.call(rl_1)];
            case 11:
                _b.sent();
                _b.label = 12;
            case 12: return [3 /*break*/, 14];
            case 13:
                if (e_1) throw e_1.error;
                return [7 /*endfinally*/];
            case 14: return [7 /*endfinally*/];
            case 15: return [2 /*return*/];
        }
    });
}); };
var mapNumToRPS = function (n) {
    switch (n) {
        case 0:
            return "R";
        case 1:
            return "P";
        case 2:
            return "S";
    }
    return "R";
};
main();
