var parseKey = require('../lib/keyParser').parseKey;

var path = require('path');
var assert = require('assert');
var inspect = require('util').inspect;
var fs = require('fs');

var EDDSA_SUPPORTED = require('../lib/constants.js').EDDSA_SUPPORTED;

function failMsg(name, message, exit) {
  var msg = '[' + name + '] ' + message;
  if (!exit)
    return msg;
  console.error(msg);
  process.exit(1);
}

fs.readdirSync(__dirname + '/fixtures').forEach(function(name) {
  if (/\.result$/i.test(name))
    return;
  if (/ed25519/i.test(name) && !EDDSA_SUPPORTED)
    return;

  var isPublic = /\.pub$/i.test(name);
  var isEncrypted = /_enc/i.test(name);
  var isPPK = /^ppk_/i.test(name);
  var key = fs.readFileSync(__dirname + '/fixtures/' + name);
  var res;
  if (isEncrypted)
    res = parseKey(key, (isPPK ? 'node.js' : 'password'));
  else
    res = parseKey(key);
  var expected = JSON.parse(
    fs.readFileSync(__dirname + '/fixtures/' + name + '.result', 'utf8')
  );
  if (typeof expected === 'string') {
    if (!(res instanceof Error))
      failMsg(name, 'Expected error: ' + expected, true);
    assert.strictEqual(expected,
                       res.message,
                       failMsg(name,
                               'Error message mismatch.\n'
                                 + 'Expected: ' + inspect(expected) + '\n'
                                 + 'Received: ' + inspect(res.message)));
  } else if (res instanceof Error) {
    failMsg(name, 'Unexpected error: ' + res.stack, true);
  } else {
    if (Array.isArray(expected) && !Array.isArray(res))
      failMsg(name, 'Expected array but did not receive one', true);
    if (!Array.isArray(expected) && Array.isArray(res))
      failMsg(name, 'Received array but did not expect one', true);

    if (!Array.isArray(res)) {
      res = [res];
      expected = [expected];
    } else if (res.length !== expected.length) {
      failMsg(name,
              'Expected ' + expected.length + ' keys, but received '
                + res.length,
              true);
    }

    res.forEach((curKey, i) => {
      var details = {
        type: curKey.type,
        comment: curKey.comment,
        public: curKey.getPublicPEM(),
        publicSSH: curKey.getPublicSSH()
                   && curKey.getPublicSSH().toString('base64'),
        private: curKey.getPrivatePEM()
      };
      assert.deepEqual(details,
                       expected[i],
                       failMsg(name,
                               'Parser output mismatch.\n'
                                 + 'Expected: ' + inspect(expected[i])
                                 + '\n\nReceived: ' + inspect(details)));
    });
  }

  if (isEncrypted && !isPublic) {
    // Make sure parsing encrypted keys without a passhprase or incorrect
    // passphrase results in an appropriate error
    var err = parseKey(key);
    if (!(err instanceof Error))
      failMsg(name, 'Expected error during parse without passphrase', true);
    if (!/no passphrase/i.test(err.message)) {
      failMsg(name,
              'Unexpected error during parse without passphrase: '
                + err.message,
              true);
    }
  }

  if (!isPublic) {
    // Try signing and verifying to make sure the private/public key PEMs are
    // correct
    var data = Buffer.from('hello world');
    res.forEach((curKey) => {
      var sig = curKey.sign(data);
      if (sig instanceof Error) {
        failMsg(name,
                'Error while signing data with key: ' + sig.message,
                true);
      }
      var verified = curKey.verify(data, sig);
      if (verified instanceof Error) {
        failMsg(name,
                'Error while verifying signed data with key: '
                  + verified.message,
                true);
      }
      if (!verified)
        failMsg(name, 'Failed to verify signed data with key', true);
    });
    if (res.length === 1 && !isPPK) {
      var pubFile = fs.readFileSync(__dirname + '/fixtures/' + name + '.pub');
      var pubParsed = parseKey(pubFile);
      if (!(pubParsed instanceof Error)) {
        var sig = res[0].sign(data);
        if (sig instanceof Error) {
          failMsg(name,
                  'Error while signing data with key: ' + sig.message,
                  true);
        }
        var verified = pubParsed.verify(data, sig);
        if (verified instanceof Error) {
          failMsg(name,
                  'Error while verifying signed data with separate public key: '
                    + verified.message,
                  true);
        }
        if (!verified) {
          failMsg(name,
                  'Failed to verify signed data with separate public key',
                  true);
        }
      }
    }
  }
});
