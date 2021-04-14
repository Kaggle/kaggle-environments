var utils = require('../lib/utils');

var path = require('path');
var assert = require('assert');

var t = -1;
var group = path.basename(__filename, '.js') + '/';

var tests = [
  { run: function() {
      var what = this.what;
      var r;

      assert.strictEqual(r = utils.readInt(Buffer.from([0,0,0]), 0),
                         false,
                         makeMsg(what, 'Wrong result: ' + r));
      next();
    },
    what: 'readInt - without stream callback - failure #1'
  },
  { run: function() {
      var what = this.what;
      var r;

      assert.strictEqual(r = utils.readInt(Buffer.from([]), 0),
                         false,
                         makeMsg(what, 'Wrong result: ' + r));
      next();
    },
    what: 'readInt - without stream callback - failure #2'
  },
  { run: function() {
      var what = this.what;
      var r;

      assert.strictEqual(r = utils.readInt(Buffer.from([0,0,0,5]), 0),
                         5,
                         makeMsg(what, 'Wrong result: ' + r));
      next();
    },
    what: 'readInt - without stream callback - success'
  },
  { run: function() {
      var what = this.what;
      var callback = function() {};
      var stream = {
        _cleanup: function(cb) {
          cleanupCalled = true;
          assert(cb === callback, makeMsg(what, 'Wrong callback'));
        }
      };
      var cleanupCalled = false;
      var r = utils.readInt(Buffer.from([]), 0, stream, callback);

      assert.strictEqual(r,
                         false,
                         makeMsg(what, 'Wrong result: ' + r));
      assert(cleanupCalled, makeMsg(what, 'Cleanup not called'));
      next();
    },
    what: 'readInt - with stream callback'
  },
  { run: function() {
      var what = this.what;
      var r;

      assert.strictEqual(r = utils.readString(Buffer.from([0,0,0]), 0),
                         false,
                         makeMsg(what, 'Wrong result: ' + r));
      next();
    },
    what: 'readString - without stream callback - bad length #1'
  },
  { run: function() {
      var what = this.what;
      var r;

      assert.strictEqual(r = utils.readString(Buffer.from([]), 0),
                         false,
                         makeMsg(what, 'Wrong result: ' + r));
      next();
    },
    what: 'readString - without stream callback - bad length #2'
  },
  { run: function() {
      var what = this.what;
      var r;

      assert.deepEqual(r = utils.readString(Buffer.from([0,0,0,1,5]), 0),
                       Buffer.from([5]),
                       makeMsg(what, 'Wrong result: ' + r));
      next();
    },
    what: 'readString - without stream callback - success'
  },
  { run: function() {
      var what = this.what;
      var r = utils.readString(Buffer.from([0,0,0,1,33]), 0, 'ascii');

      assert.deepEqual(r,
                       '!',
                       makeMsg(what, 'Wrong result: ' + r));
      next();
    },
    what: 'readString - without stream callback - encoding'
  },
  { run: function() {
      var what = this.what;
      var callback = function() {};
      var stream = {
        _cleanup: function(cb) {
          cleanupCalled = true;
          assert(cb === callback, makeMsg(what, 'Wrong callback'));
        }
      };
      var cleanupCalled = false;
      var r;

      assert.deepEqual(r = utils.readString(Buffer.from([0,0,0,1]),
                                            0,
                                            stream,
                                            callback),
                       false,
                       makeMsg(what, 'Wrong result: ' + r));
      assert(cleanupCalled, makeMsg(what, 'Cleanup not called'));
      next();
    },
    what: 'readString - with stream callback - no encoding'
  },
  { run: function() {
      var what = this.what;
      var callback = function() {};
      var stream = {
        _cleanup: function(cb) {
          cleanupCalled = true;
          assert(cb === callback, makeMsg(what, 'Wrong callback'));
        }
      };
      var cleanupCalled = false;
      var r;

      assert.deepEqual(r = utils.readString(Buffer.from([0,0,0,1]),
                                            0,
                                            'ascii',
                                            stream,
                                            callback),
                       false,
                       makeMsg(what, 'Wrong result: ' + r));
      assert(cleanupCalled, makeMsg(what, 'Cleanup not called'));
      next();
    },
    what: 'readString - with stream callback - encoding'
  },
];

function next() {
  if (Array.isArray(process._events.exit))
    process._events.exit = process._events.exit[1];
  if (++t === tests.length)
    return;

  var v = tests[t];
  process.nextTick(function() {
    v.run.call(v);
  });
}

function makeMsg(what, msg) {
  return '[' + group + what + ']: ' + msg;
}

process.once('exit', function() {
  assert(t === tests.length,
         makeMsg('_exit',
                 'Only finished ' + t + '/' + tests.length + ' tests'));
});

next();
