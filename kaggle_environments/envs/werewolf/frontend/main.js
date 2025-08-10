import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js';

import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js';
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

    document.body.appendChild(this._threejs.domElement);

    window.addEventListener('resize', () => {
      this._OnWindowResize();
    }, false);

    const fov = 60;
    const aspect = window.innerWidth / window.innerHeight;
    const near = 1.0;
    const far = 100000.0;
    this._camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    this._camera.position.set(0, 0, 50); // Initial position, will be framed later

    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x7393B3);

    const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 1.5);
    this._scene.add(hemisphereLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight.position.set(20, 50, 10);
    dirLight.castShadow = true;
    this._scene.add(dirLight);

    this._controls = new OrbitControls(this._camera, this._threejs.domElement);
    this._controls.target.set(0, 0, 0);
    this._controls.update();

    this._LoadModels();
    this._RAF();
  }

  _LoadModels() {
    const fbxLoader = new FBXLoader();
    fbxLoader.load('assets/Idle_stickman.fbx', (fbx) => {
        const stickmanModel = this._NormalizeModel(fbx);
        const stickmanGroup = new THREE.Group();
        const idleClip = fbx.animations[0];

        if (!idleClip) {
            console.error("FBX file does not contain an animation clip.");
        }

        const numStickmen = 8;
        const radius = 15;
        const sectorAngleDegrees = 60;
        const sectorAngleRadians = sectorAngleDegrees * (Math.PI / 180);
        const stickmanHeight = 4;

        const startAngle = -sectorAngleRadians / 2;
        const angleIncrement = numStickmen > 1 ? sectorAngleRadians / (numStickmen - 1) : 0;

        for (let i = 0; i < numStickmen; i++) {
          const stickman = SkeletonUtils.clone(stickmanModel);
          const angle = startAngle + i * angleIncrement;

          const x = radius * Math.sin(angle);
          const z = radius * Math.cos(angle);
          const y = stickmanHeight / 2;
          stickman.position.set(x, y, z);

          stickman.scale.multiplyScalar(stickmanHeight);
          
          // Make stickman face the center of the circle (at its own height)
          stickman.lookAt(new THREE.Vector3(0, y, 0));

          stickmanGroup.add(stickman);

          // Set a static pose if animation exists
          if (idleClip) {
            const mixer = new THREE.AnimationMixer(stickman);
            const action = mixer.clipAction(idleClip);
            action.time = Math.random() * idleClip.duration; // Set random frame
            action.play();
            mixer.update(0); // Update once to apply the pose
          }
        }
        
        this._scene.add(stickmanGroup);
        this._FrameGroup(stickmanGroup);
      },
      undefined,
      (error) => {
        console.error("Error loading FBX model:", error);
      }
    );
  }

  _FrameGroup(group) {
    const box = new THREE.Box3().setFromObject(group);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this._camera.fov * (Math.PI / 180);
    
    // Calculate distance to fit the group and then pull back slightly
    let cameraZ = Math.abs(maxDim / Math.tan(fov / 2));
    
    // Adjust this multiplier to be closer/further. <1 is closer.
    cameraZ *= 0.8;

    // Position camera in front of the group, along the Z-axis
    this._camera.position.set(center.x, center.y, -cameraZ);

    // Shift camera and target up to place group in bottom half
    // A smaller divisor here makes the group take up more of the bottom half
    const shiftY = size.y / 2.5; 
    this._camera.position.y += shiftY;

    const newTarget = center.clone();
    newTarget.y += shiftY;
    this._controls.target.copy(newTarget);
    this._controls.update();
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
      this._threejs.render(this._scene, this._camera);
      this._RAF();
    });
  }
}


let _APP = null;

window.addEventListener('DOMContentLoaded', () => {
  _APP = new BasicWorldDemo();
});
