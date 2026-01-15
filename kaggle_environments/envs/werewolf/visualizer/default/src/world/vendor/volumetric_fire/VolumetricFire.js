import * as THREE from 'three';

// VITE MAGIC: Import images directly.
// Vite will swap these variables for the final public URL (e.g. /assets/nzw.123hash.png)
// matching your GCS base path automatically.
import nzwUrl from './nzw.png';
import firetexUrl from './firetex.png';

/**
 * @author yomotsu / http://yomotsu.net
 * Modernized for Vite/ESM
 */

const vs = `
  attribute vec3 position;
  attribute vec3 tex;
  uniform mat4 projectionMatrix;
  uniform mat4 modelViewMatrix;
  varying vec3 texOut;
  void main ( void ) {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0 );
    texOut = tex;
  }
`;

const fs = `
  precision highp float;
  vec2 mBBS( vec2 val, float modulus ) {
    val = mod( val, modulus ); 
    return mod(val * val, modulus);
  }
  uniform sampler2D nzw;
  const float modulus = 61.0; 
  float mnoise ( vec3 pos ) {
    float intArg = floor( pos.z );
    float fracArg = fract( pos.z );
    vec2 hash = mBBS( intArg * 3.0 + vec2( 0, 3 ), modulus );
    vec4 g = vec4 (
      texture2D( nzw, vec2( pos.x, pos.y + hash.x ) / modulus ).xy,
      texture2D( nzw, vec2( pos.x, pos.y + hash.y ) / modulus ).xy
    ) * 2.0 - 1.0;
    return mix(
      g.x + g.y * fracArg,
      g.z + g.w * ( fracArg - 1.0 ),
      smoothstep( 0.0, 1.0, fracArg )
    );
  }
  const int octives = 2;
  const float lacunarity = 2.0;
  const float gain = 0.5;
  float turbulence( vec3 pos ) {
    float sum  = 0.0;
    float freq = 1.0;
    float amp  = 1.0;
    for ( int i = 0; i < 2; i++ ) {
      sum += abs( mnoise( pos * freq ) ) * amp;
      freq *= lacunarity;
      amp *= gain;
    }
    return sum;
  }
  const float magnatude = 1.3;
  uniform float time;
  uniform sampler2D fireProfile;
  vec4 sampleFire( vec3 loc, vec4 scale ) {
    loc.xz = loc.xz * 2.0 - 1.0;
    vec2 st = vec2( sqrt( dot( loc.xz, loc.xz ) ), loc.y );
    loc.y -= time * scale.w; 
    loc *= scale.xyz; 
    float offset = sqrt( st.y ) * magnatude * turbulence( loc );
    st.y += offset;
    if ( st.y > 1.0 ) {
      return vec4( 0, 0, 0, 1 );
    }
    vec4 result = texture2D( fireProfile, st );
    if ( st.y < 0.1 ) {
      result *= st.y / 0.1;
    }
    return result;
  }
  varying vec3 texOut;
  void main( void ) {
    vec3 color = sampleFire( texOut, vec4( 1.0, 2.0, 1.0, 0.5 ) ).xyz;
    gl_FragColor = vec4( color * 1.5, 1 );
  }
`;

// Helper Class
class PriorityQueue {
  constructor() {
    this.contents = [];
    this.sorted = false;
  }
  sort() {
    this.contents.sort((a, b) => a.priority - b.priority);
    this.sorted = true;
  }
  pop() {
    if (!this.sorted) this.sort();
    return this.contents.pop();
  }
  top() {
    if (!this.sorted) this.sort();
    return this.contents[this.contents.length - 1];
  }
  push(object, priority) {
    this.contents.push({ object: object, priority: priority });
    this.sorted = false;
  }
}

// Singleton for material generation
const initMaterial = (function () {
  let material;
  const textureLoader = new THREE.TextureLoader();

  return function () {
    // eslint-disable-next-line no-extra-boolean-cast
    if (!!material) {
      return material;
    }

    // Use the imported URLs here!
    const nzw = textureLoader.load(nzwUrl);
    nzw.wrapS = THREE.RepeatWrapping;
    nzw.wrapT = THREE.RepeatWrapping;
    nzw.magFilter = THREE.LinearFilter;
    nzw.minFilter = THREE.LinearFilter;

    const fireProfile = textureLoader.load(firetexUrl);
    fireProfile.wrapS = THREE.ClampToEdgeWrapping;
    fireProfile.wrapT = THREE.ClampToEdgeWrapping;
    fireProfile.magFilter = THREE.LinearFilter;
    fireProfile.minFilter = THREE.LinearFilter;

    const uniforms = {
      nzw: { type: 't', value: nzw },
      fireProfile: { type: 't', value: fireProfile },
      time: { type: 'f', value: 1.0 },
    };

    material = new THREE.RawShaderMaterial({
      vertexShader: vs,
      fragmentShader: fs,
      uniforms: uniforms,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
    });

    return material;
  };
})();

const cornerNeighbors = [
  [1, 2, 4],
  [0, 5, 3],
  [0, 3, 6],
  [1, 7, 2],
  [0, 6, 5],
  [1, 4, 7],
  [2, 7, 4],
  [3, 5, 6],
];

const incomingEdges = [
  [-1, 2, 4, -1, 1, -1, -1, -1],
  [5, -1, -1, 0, -1, 3, -1, -1],
  [3, -1, -1, 6, -1, -1, 0, -1],
  [-1, 7, 1, -1, -1, -1, -1, 2],
  [6, -1, -1, -1, -1, 0, 5, -1],
  [-1, 4, -1, -1, 7, -1, -1, 1],
  [-1, -1, 7, -1, 2, -1, -1, 4],
  [-1, -1, -1, 5, -1, 6, 3, -1],
];

// Export as a standard ES Class
export default class VolumetricFire {
  constructor(width, height, depth, sliceSpacing, camera) {
    this.camera = camera;
    this._sliceSpacing = sliceSpacing;

    const widthHalf = width * 0.5;
    const heightHalf = height * 0.5;
    const depthHalf = depth * 0.5;

    this._posCorners = [
      new THREE.Vector3(-widthHalf, -heightHalf, -depthHalf),
      new THREE.Vector3(widthHalf, -heightHalf, -depthHalf),
      new THREE.Vector3(-widthHalf, heightHalf, -depthHalf),
      new THREE.Vector3(widthHalf, heightHalf, -depthHalf),
      new THREE.Vector3(-widthHalf, -heightHalf, depthHalf),
      new THREE.Vector3(widthHalf, -heightHalf, depthHalf),
      new THREE.Vector3(-widthHalf, heightHalf, depthHalf),
      new THREE.Vector3(widthHalf, heightHalf, depthHalf),
    ];
    this._texCorners = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(1, 1, 0),
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(1, 0, 1),
      new THREE.Vector3(0, 1, 1),
      new THREE.Vector3(1, 1, 1),
    ];

    this._viewVector = new THREE.Vector3();

    const index = new Uint16Array((width + height + depth) * 30);
    const position = new Float32Array((width + height + depth) * 30 * 3);
    const tex = new Float32Array((width + height + depth) * 30 * 3);

    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
    geometry.setAttribute('tex', new THREE.BufferAttribute(tex, 3));

    const material = initMaterial();

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    this.mesh.updateMatrixWorld();
  }

  update(elapsed) {
    this.updateViewVector();
    this.slice();
    this.updateGeometry();
    this.mesh.material.uniforms.time.value = elapsed;
  }

  updateGeometry() {
    this.mesh.geometry.index.array.set(this._indexes);
    this.mesh.geometry.attributes.position.array.set(this._points);
    this.mesh.geometry.attributes.tex.array.set(this._texCoords);

    this.mesh.geometry.index.needsUpdate = true;
    this.mesh.geometry.attributes.position.needsUpdate = true;
    this.mesh.geometry.attributes.tex.needsUpdate = true;
  }

  updateViewVector() {
    const modelViewMatrix = new THREE.Matrix4();
    modelViewMatrix.multiplyMatrices(this.camera.matrixWorldInverse, this.mesh.matrixWorld);

    this._viewVector
      .set(-modelViewMatrix.elements[2], -modelViewMatrix.elements[6], -modelViewMatrix.elements[10])
      .normalize();
  }

  slice() {
    this._points = [];
    this._texCoords = [];
    this._indexes = [];

    let i;
    const cornerDistance0 = this._posCorners[0].dot(this._viewVector);
    const cornerDistance = [cornerDistance0];
    let maxCorner = 0;
    let minDistance = cornerDistance0;
    let maxDistance = cornerDistance0;

    for (i = 1; i < 8; i = (i + 1) | 0) {
      cornerDistance[i] = this._posCorners[i].dot(this._viewVector);
      if (cornerDistance[i] > maxDistance) {
        maxCorner = i;
        maxDistance = cornerDistance[i];
      }
      if (cornerDistance[i] < minDistance) {
        minDistance = cornerDistance[i];
      }
    }

    let sliceDistance = Math.floor(maxDistance / this._sliceSpacing) * this._sliceSpacing;

    const activeEdges = [];
    let firstEdge = 0;
    let nextEdge = 0;
    const expirations = new PriorityQueue();

    const createEdge = (startIndex, endIndex) => {
      if (nextEdge >= 12) {
        return undefined;
      }

      const activeEdge = {
        expired: false,
        startIndex: startIndex,
        endIndex: endIndex,
        deltaPos: new THREE.Vector3(),
        deltaTex: new THREE.Vector3(),
        pos: new THREE.Vector3(),
        tex: new THREE.Vector3(),
        cur: nextEdge,
      };

      const range = cornerDistance[startIndex] - cornerDistance[endIndex];

      if (range !== 0.0) {
        const irange = 1.0 / range;
        activeEdge.deltaPos.subVectors(this._posCorners[endIndex], this._posCorners[startIndex]).multiplyScalar(irange);

        activeEdge.deltaTex.subVectors(this._texCorners[endIndex], this._texCorners[startIndex]).multiplyScalar(irange);

        const step = cornerDistance[startIndex] - sliceDistance;

        activeEdge.pos.addVectors(activeEdge.deltaPos.clone().multiplyScalar(step), this._posCorners[startIndex]);

        activeEdge.tex.addVectors(activeEdge.deltaTex.clone().multiplyScalar(step), this._texCorners[startIndex]);

        activeEdge.deltaPos.multiplyScalar(this._sliceSpacing);
        activeEdge.deltaTex.multiplyScalar(this._sliceSpacing);
      }

      expirations.push(activeEdge, cornerDistance[endIndex]);
      activeEdges[nextEdge++] = activeEdge;
      return activeEdge;
    };

    for (i = 0; i < 3; i = (i + 1) | 0) {
      const activeEdge = createEdge(maxCorner, cornerNeighbors[maxCorner][i]);
      activeEdge.prev = (i + 2) % 3;
      activeEdge.next = (i + 1) % 3;
    }

    let nextIndex = 0;

    while (sliceDistance > minDistance) {
      while (expirations.top().priority >= sliceDistance) {
        const edge = expirations.pop().object;
        if (edge.expired) {
          continue;
        }

        if (edge.endIndex !== activeEdges[edge.prev].endIndex && edge.endIndex !== activeEdges[edge.next].endIndex) {
          edge.expired = true;
          const activeEdge1 = createEdge(edge.endIndex, incomingEdges[edge.endIndex][edge.startIndex]);
          activeEdge1.prev = edge.prev;
          activeEdges[edge.prev].next = nextEdge - 1;
          activeEdge1.next = nextEdge;

          const activeEdge2 = createEdge(edge.endIndex, incomingEdges[edge.endIndex][activeEdge1.endIndex]);
          activeEdge2.prev = nextEdge - 2;
          activeEdge2.next = edge.next;
          activeEdges[activeEdge2.next].prev = nextEdge - 1;
          firstEdge = nextEdge - 1;
        } else {
          let prev, next;
          if (edge.endIndex === activeEdges[edge.prev].endIndex) {
            prev = activeEdges[edge.prev];
            next = edge;
          } else {
            prev = edge;
            next = activeEdges[edge.next];
          }

          prev.expired = true;
          next.expired = true;

          const activeEdge = createEdge(edge.endIndex, incomingEdges[edge.endIndex][prev.startIndex]);
          activeEdge.prev = prev.prev;
          activeEdges[activeEdge.prev].next = nextEdge - 1;
          activeEdge.next = next.next;
          activeEdges[activeEdge.next].prev = nextEdge - 1;
          firstEdge = nextEdge - 1;
        }
      }

      let cur = firstEdge;
      let count = 0;

      do {
        ++count;
        const activeEdge = activeEdges[cur];
        this._points.push(activeEdge.pos.x, activeEdge.pos.y, activeEdge.pos.z);
        this._texCoords.push(activeEdge.tex.x, activeEdge.tex.y, activeEdge.tex.z);
        activeEdge.pos.add(activeEdge.deltaPos);
        activeEdge.tex.add(activeEdge.deltaTex);
        cur = activeEdge.next;
      } while (cur !== firstEdge);

      for (i = 2; i < count; i = (i + 1) | 0) {
        this._indexes.push(nextIndex, nextIndex + i - 1, nextIndex + i);
      }

      nextIndex += count;
      sliceDistance -= this._sliceSpacing;
    }
  }
}
