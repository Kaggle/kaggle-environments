import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js';

import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/GLTFLoader.js';
import {FBXLoader} from "https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/FBXLoader.js";
import { SkeletonUtils } from 'https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/utils/SkeletonUtils.js';


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

    this._clock = new THREE.Clock();
    this._mixers = [];

    document.body.appendChild(this._threejs.domElement);

    window.addEventListener('resize', () => {
      this._OnWindowResize();
    }, false);

    const fov = 60;
    const aspect = window.innerWidth / 1080;
    const near = 1.0;
    const far = 100000.0;
    this._camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    this._camera.position.set(0, 10, -25); // Moved camera to the front

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
    const fbxLoader = new FBXLoader();

    fbxLoader.load('assets/Idle_stickman.fbx', (fbx) => {
        const stickmanModel = this._NormalizeModel(fbx);
        const idleClip = fbx.animations[0];

        if (!idleClip) {
            console.error("FBX file does not contain an animation clip.");
            return;
        }

        const numStickmen = 8;
        const radius = 15;
        const sectorAngleDegrees = 60;
        const sectorAngleRadians = sectorAngleDegrees * (Math.PI / 180);
        const stickmanHeight = 4;

        const startAngle = -sectorAngleRadians / 2;
        const angleIncrement = numStickmen > 1 ? sectorAngleRadians / (numStickmen - 1) : 0;

        for (let i = 0; i < numStickmen; i++) {
          // We need to clone the model for each instance using SkeletonUtils
          const stickman = SkeletonUtils.clone(stickmanModel);
          
          const angle = startAngle + i * angleIncrement;

          const x = radius * Math.sin(angle);
          const z = radius * Math.cos(angle);
          const y = stickmanHeight / 2;
          stickman.position.set(x, y, z);

          stickman.scale.multiplyScalar(stickmanHeight);
          stickman.lookAt(new THREE.Vector3(0, y, 0));

          this._scene.add(stickman);

          // Animation
          const mixer = new THREE.AnimationMixer(stickman);
          const idleAction = mixer.clipAction(idleClip);
          idleAction.play();
          this._mixers.push(mixer);
        }
      },
      undefined,
      (error) => {
        console.error("Error loading FBX model:", error);
      }
    );
  }

  _NormalizeModel(model) {
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);
    const scale = 1.0 / size.y;
    model.scale.set(scale, scale, scale);
    return model;
  }

  _OnWindowResize() {
    this._camera.aspect = window.innerWidth / window.innerHeight;
    this._camera.updateProjectionMatrix();
    this._threejs.setSize(window.innerWidth, window.innerHeight);
  }

  _RAF() {
    requestAnimationFrame(() => {
      const delta = this._clock.getDelta();
      for (const mixer of this._mixers) {
        mixer.update(delta);
      }
      this._threejs.render(this._scene, this._camera);
      this._RAF();
    });
  }
}


let _APP = null;

window.addEventListener('DOMContentLoaded', () => {
  _APP = new BasicWorldDemo();
});
