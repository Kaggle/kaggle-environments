var SSH2Stream = require('../lib/ssh');
var parseKey = require('../lib/utils').parseKey;
var MESSAGE = require('../lib/constants').MESSAGE;

var assert = require('assert');
var fs = require('fs');

var SERVER_KEY = fs.readFileSync(__dirname + '/fixtures/openssh_new_rsa');
var SERVER_KEY_PRV = parseKey(SERVER_KEY);

var server = new SSH2Stream({
  server: true,
  hostKeys: {
    'ssh-rsa': SERVER_KEY_PRV
  },
  algorithms: {
    serverHostKey: ['ssh-rsa']
  }
});
var client = new SSH2Stream();
var cliError;
var srvError;

server.on('error', function(err) {
  assert(err);
  assert(/unexpected/.test(err.message));
  assert(!srvError);
  srvError = err;
});

// Removed 'KEXDH_REPLY' listeners as it causes client to send 'NEWKEYS' which
// changes server's state.
client.removeAllListeners('KEXDH_REPLY');
// Removed 'NEWKEYS' listeners as server sends 'NEWKEYS' after receiving
// 'KEXDH_INIT' which causes errors on client if 'NEWKEYS' is processed
// without processing 'KEXDH_REPLY'
client.removeAllListeners('NEWKEYS');
// Added 'KEXDH_REPLY' which violates protocol and re-sends 'KEXDH_INIT'
// packet
client.on('KEXDH_REPLY', function(info) {
  var state = client._state;
  var outstate = state.outgoing;
  var buf = Buffer.allocUnsafe(1 + 4 + outstate.pubkey.length);
  buf[0] = MESSAGE.KEXDH_INIT;
  buf.writeUInt32BE(outstate.pubkey.length, 1, true);
  outstate.pubkey.copy(buf, 5);
  SSH2Stream._send(client, buf, undefined, true);
});
client.on('error', function(err) {
  assert(!cliError);
  assert(err);
  assert.equal(
    err.message,
    'PROTOCOL_ERROR',
    'Expected Error: PROTOCOL_ERROR Got Error: ' + err.message
  );
  cliError = err;
});
client.pipe(server).pipe(client);

process.on('exit', function() {
  assert(cliError, 'Expected client error');
});
