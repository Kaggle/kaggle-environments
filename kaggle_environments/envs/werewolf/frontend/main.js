import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js';

import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/GLTFLoader.js';


class BasicWorldDemo {
  constructor() {
    this._Initialize();
  }

  _Initialize() {
    this._threejs = new THREE.WebGLRenderer({
      antialias: true,
    });
    this._threejs.shadowMap.enabled = true;
    this._threejs.shadowMap.type = THREE.PCFSoftShadowMap;
    this._threejs.setPixelRatio(window.devicePixelRatio);
    this._threejs.setSize(window.innerWidth, window.innerHeight);

    document.body.appendChild(this._threejs.domElement);

    window.addEventListener('resize', () => {
      this._OnWindowResize();
    }, false);

    const fov = 60;
    const aspect = 1920 / 1080;
    const near = 1.0;
    const far = 100000.0;
    this._camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    this._camera.position.set(0, 5, 25);

    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x7393B3); // A calm blue-grey background

    // Add a hemisphere light for soft ambient lighting
    const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 1.5);
    this._scene.add(hemisphereLight);

    // Add a stronger directional light for highlights and shadows
    const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight.position.set(20, 50, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    this._scene.add(dirLight);

    const controls = new OrbitControls(
      this._camera, this._threejs.domElement);
    controls.target.set(0, 0, 0); // Target the world origin
    controls.update();

    this._LoadModels();
    this._RAF();
  }

  _LoadModels() {
    const loader = new GLTFLoader();

    const loadPromises = [
      this._LoadAndProcessModel(loader, '../../../assets/wolf-kun/scene.gltf'),
      this._LoadAndProcessModel(loader, '../../../assets/low_poly_medieval_windmill/scene.gltf')
    ];

    Promise.all(loadPromises).then(([wolfModel, windmillModel]) => {
      // Now that both models are loaded and normalized to a height of 1...

      // 1. Create clones
      const wolf1 = wolfModel;
      const wolf2 = wolfModel.clone();
      const windmill = windmillModel;

      // 2. Set final scales
      const wolfHeight = 4;
      const windmillHeight = wolfHeight * 2;

      wolf1.scale.multiplyScalar(wolfHeight);
      wolf2.scale.multiplyScalar(wolfHeight);
      windmill.scale.multiplyScalar(windmillHeight);

      // 3. Position them on the same horizontal plane (y=0) with wolves in front
      const wolfSpacing = 5; // Spacing between the two wolves
      const windmillZ = 0;
      const wolfZ = 5;

      // To place the base on y=0, we lift the model by half its final height
//      windmill.position.set(0, windmillHeight / 2, windmillZ);
//      wolf1.position.set(-wolfSpacing / 2, wolfHeight / 2, wolfZ);
//      wolf2.position.set(wolfSpacing / 2, wolfHeight / 2, wolfZ);
      windmill.position.set(-5, - windmillHeight / 2, 0);
      wolf1.position.set(0, - wolfHeight / 2, 0)
      wolf2.position.set(3, - wolfHeight / 2, 0)


      // 4. Add to the scene
      this._scene.add(windmill);
      this._scene.add(wolf1);
      this._scene.add(wolf2);

    }).catch(error => {
      console.error("Error loading models:", error);
    });
  }

  _LoadAndProcessModel(loader, url) {
    return new Promise((resolve, reject) => {
      loader.load(url,
        (gltf) => {
          const model = gltf.scene;

          // --- Normalize logic ---
          const box = new THREE.Box3().setFromObject(model);
          const size = box.getSize(new THREE.Vector3());
          const center = box.getCenter(new THREE.Vector3());

          // Center the model
          model.position.sub(center);

          // Scale model to a height of 1
          const scale = 1.0 / size.y;
          model.scale.set(scale, scale, scale);

          resolve(model);
        },
        undefined,
        (error) => {
          console.error(`An error occurred loading ${url}`, error);
          reject(error);
        }
      );
    });
  }

  _OnWindowResize() {
    this._camera.aspect = window.innerWidth / window.innerHeight;
    this._camera.updateProjectionMatrix();
    this._threejs.setSize(window.innerWidth, window.innerHeight);
  }

  _RAF() {
    requestAnimationFrame(() => {
      this._threejs.render(this._scene, this._camera);
      this._RAF();
    });
  }
}


let _APP = null;

window.addEventListener('DOMContentLoaded', () => {
  _APP = new BasicWorldDemo();
});