var assert = require('assert');
var http = require('http');
var Modem = require('../lib/modem');
var defaultSocketPath = require('os').type() === 'Windows_NT' ? '//./pipe/docker_engine' : '/var/run/docker.sock';

describe('Modem', function () {
  beforeEach(function () {
    delete process.env.DOCKER_HOST;
  });

  it('should default to default socket path', function () {
    var modem = new Modem();
    assert.ok(modem.socketPath);
    assert.strictEqual(modem.socketPath, defaultSocketPath);
  });

  it('should use specific cert, key and ca', function () {
    var ca = 'caaaaa';
    var cert = 'certtttt';
    var key = 'keyyyyy';
    var modem = new Modem({
      version: 'v1.39',
      host: '127.0.0.1',
      port: 2376,
      ca,
      cert,
      key
    });

    assert.strictEqual(ca, modem.ca);
    assert.strictEqual(cert, modem.cert);
    assert.strictEqual(key, modem.key);
  });

  it('shouldn\'t default to default socket path', function () {
    var modem = new Modem({
      protocol: 'http',
      host: '127.0.0.1',
      port: 2375
    });
    assert.ok(modem.host);
    assert.strictEqual(modem.socketPath, undefined);
  });

  it('should use default value if argument not defined in constructor parameter object', function () {
    var customHeaders = {
      host: 'my-custom-host'
    };
    var modem = new Modem({
      headers: customHeaders
    });
    assert.ok(modem.headers);
    assert.ok(modem.socketPath);
    assert.strictEqual(modem.socketPath, defaultSocketPath);
    assert.strictEqual(modem.headers, customHeaders);
  });

  it('should allow DOCKER_HOST=unix:///path/to/docker.sock', function () {
    process.env.DOCKER_HOST = 'unix:///tmp/docker.sock';

    var modem = new Modem();
    assert.ok(modem.socketPath);
    assert.strictEqual(modem.socketPath, '/tmp/docker.sock');
  });

  it('should interpret DOCKER_HOST=unix:// as /var/run/docker.sock', function () {
    process.env.DOCKER_HOST = 'unix://';

    var modem = new Modem();
    assert.ok(modem.socketPath);
    assert.strictEqual(modem.socketPath, '/var/run/docker.sock');
  });

  it('should interpret DOCKER_HOST=tcp://N.N.N.N:2376 as https', function () {
    process.env.DOCKER_HOST = 'tcp://192.168.59.103:2376';

    var modem = new Modem();
    assert.ok(modem.host);
    assert.ok(modem.port);
    assert.ok(modem.protocol);
    assert.strictEqual(modem.host, '192.168.59.103');
    assert.strictEqual(modem.port, '2376');
    assert.strictEqual(modem.protocol, 'https');
  });

  it('should interpret DOCKER_HOST=tcp://[::1]:12345', function () {
    process.env.DOCKER_HOST = 'tcp://[::1]:12345';

    var modem = new Modem();
    assert.ok(modem.host);
    assert.ok(modem.port);
    assert.ok(modem.protocol);
    assert.strictEqual(modem.host, '[::1]');
    assert.strictEqual(modem.port, '12345');
    assert.strictEqual(modem.protocol, 'http');
  });

  it('should interpret DOCKER_HOST=tcp://N.N.N.N:5555 as http', function () {
    delete process.env.DOCKER_TLS_VERIFY;
    process.env.DOCKER_HOST = 'tcp://192.168.59.105:5555';

    var modem = new Modem();
    assert.ok(modem.host);
    assert.ok(modem.port);
    assert.ok(modem.protocol);
    assert.strictEqual(modem.host, '192.168.59.105');
    assert.strictEqual(modem.port, '5555');
    assert.strictEqual(modem.protocol, 'http');
  });

  it('should interpret DOCKER_HOST=tcp://N.N.N.N:5555 as http', function () {
    process.env.DOCKER_TLS_VERIFY = '1';
    process.env.DOCKER_HOST = 'tcp://192.168.59.105:5555';

    var modem = new Modem();
    assert.ok(modem.host);
    assert.ok(modem.port);
    assert.ok(modem.protocol);
    assert.strictEqual(modem.host, '192.168.59.105');
    assert.strictEqual(modem.port, '5555');
    assert.strictEqual(modem.protocol, 'https');
  });

  it('should accept DOCKER_HOST=N.N.N.N:5555 as http', function () {
    delete process.env.DOCKER_TLS_VERIFY;
    process.env.DOCKER_HOST = '192.168.59.105:5555';

    var modem = new Modem();
    assert.ok(modem.host);
    assert.ok(modem.port);
    assert.ok(modem.protocol);
    assert.strictEqual(modem.host, '192.168.59.105');
    assert.strictEqual(modem.port, '5555');
    assert.strictEqual(modem.protocol, 'http');
  });

  it('should auto encode querystring option maps as JSON', function () {
    var modem = new Modem();

    var opts = {
      "limit": 12,
      "filters": {
        "label": ["staging", "env=green"]
      },
      "t": ["repo:latest", "repo:1.0.0"]
    };
    var control = 'limit=12&filters={"label"%3A["staging"%2C"env%3Dgreen"]}&t=repo%3Alatest&t=repo%3A1.0.0';
    var qs = modem.buildQuerystring(opts);
    assert.strictEqual(decodeURI(qs), control);
  });

  it('should parse DOCKER_CLIENT_TIMEOUT from environment', function () {
    process.env.DOCKER_HOST = '192.168.59.105:5555';
    process.env.DOCKER_CLIENT_TIMEOUT = 3000;

    var modem = new Modem();
    assert.strictEqual(modem.timeout, 3000);
  });

  it('should use default http agent when no agent is specified', function () {
    var modem = new Modem();
    assert.strictEqual(typeof modem.agent, 'undefined');
  });

  it('should use custom http agent', function () {
    var httpAgent = new http.Agent({
      keepAlive: true
    });
    var modem = new Modem({
      agent: httpAgent
    });
    assert.ok(modem.agent instanceof http.Agent);
    assert.strictEqual(modem.agent, httpAgent);
  });
});
